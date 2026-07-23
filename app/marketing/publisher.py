"""Meta (Facebook/Instagram) publishing.

Defaults to dry-run. This is the one component in the POC that can affect the outside
world: a live call posts publicly under a real bakery's brand and cannot be quietly
undone, since followers may have already seen it.

So live publishing requires three separate things to line up -- credentials present,
`enabled` switched on, and `dry_run=false` on the request. Any missing one yields a
preview. That is not defensive coding for its own sake; the brief asked for posting
"with no human intervention", and the safe way to build towards that is to make the
automatic path fully functional while keeping the trigger explicit until the owner
has watched it work.
"""

from __future__ import annotations

import os

GRAPH_API_VERSION = "v21.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


class MetaPublisher:
    """Thin Meta Graph API client with a preview mode."""

    def __init__(
        self,
        page_id: str | None = None,
        ig_user_id: str | None = None,
        access_token: str | None = None,
        enabled: bool | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.page_id = page_id or os.getenv("META_PAGE_ID")
        self.ig_user_id = ig_user_id or os.getenv("META_IG_USER_ID")
        self.access_token = access_token or os.getenv("META_ACCESS_TOKEN")
        # Separate kill switch, so credentials being present is never sufficient.
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("META_PUBLISH_ENABLED", "false").lower() == "true"
        )
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.page_id and self.access_token)

    def _render_preview(self, sku: str, copy_ar: str, platforms: list[str]) -> dict:
        return {
            platform: {
                "text": copy_ar,
                "target": self.page_id if platform == "facebook" else self.ig_user_id,
                "character_count": len(copy_ar),
            }
            for platform in platforms
        }

    def _post_facebook(self, copy_ar: str) -> str:
        import httpx

        response = httpx.post(
            f"{GRAPH_BASE}/{self.page_id}/feed",
            data={"message": copy_ar, "access_token": self.access_token},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("id", "")

    def publish(
        self, sku: str, copy_ar: str, platforms: list[str], dry_run: bool = True,
    ) -> dict:
        preview = self._render_preview(sku, copy_ar, platforms)

        if dry_run:
            return {
                "status": "preview",
                "dry_run": True,
                "platforms": platforms,
                "preview": preview,
                "post_ids": {},
                "message": "Preview only -- nothing was sent to Meta.",
            }

        if not self.configured:
            return {
                "status": "failed",
                "dry_run": False,
                "platforms": platforms,
                "preview": preview,
                "post_ids": {},
                "message": (
                    "Meta credentials are not configured. Set META_PAGE_ID and "
                    "META_ACCESS_TOKEN to publish."
                ),
            }

        if not self.enabled:
            return {
                "status": "failed",
                "dry_run": False,
                "platforms": platforms,
                "preview": preview,
                "post_ids": {},
                "message": (
                    "Live publishing is disabled. Set META_PUBLISH_ENABLED=true to "
                    "allow posting to the live page."
                ),
            }

        post_ids: dict[str, str] = {}
        errors: list[str] = []
        for platform in platforms:
            try:
                if platform == "facebook":
                    post_ids[platform] = self._post_facebook(copy_ar)
                else:
                    # Instagram requires a media container (image) before publishing;
                    # text-only posts are not supported by the Graph API.
                    errors.append(
                        "instagram: not implemented -- requires an image asset pipeline"
                    )
            except Exception as exc:
                errors.append(f"{platform}: {exc}")

        return {
            "status": "published" if post_ids else "failed",
            "dry_run": False,
            "platforms": platforms,
            "preview": preview,
            "post_ids": post_ids,
            "message": "; ".join(errors) if errors else "Published successfully.",
        }
