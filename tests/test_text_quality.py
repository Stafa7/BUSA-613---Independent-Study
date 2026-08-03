from health_misinfo.features.text import (
    canonical_article_url,
    content_exclusion_reason,
    mask_artifacts,
)


def test_content_quality_rejects_known_scraper_pages():
    assert content_exclusion_reason('""') == "near_empty_text"
    assert content_exclusion_reason("Access denied by Cloudflare") == "access_denied_page"
    assert content_exclusion_reason("403 Forbidden: you do not have permission") == "forbidden_page"
    assert content_exclusion_reason("Log In or Sign Up to View this post") == "login_or_signup_page"
    assert (
        content_exclusion_reason(
            "Please make sure your browser supports JavaScript and cookies and that you are not blocking them."
        )
        == "javascript_disabled_page"
    )


def test_canonical_url_preserves_identity_parameters_but_removes_tracking():
    first = canonical_article_url("http://example.com/article?id=10&utm_source=test")
    second = canonical_article_url("https://www.example.com/article?id=11&utm_source=test")
    assert first == "https://example.com/article?id=10"
    assert second == "https://example.com/article?id=11"
    assert first != second


def test_masking_removes_source_platform_and_label_terms():
    masked = mask_artifacts(
        "Facebook says this fake report came from example.com",
        "example.com",
    )
    assert "facebook" not in masked.lower()
    assert "fake" not in masked.lower()
    assert "example" not in masked.lower()
    assert "[source]" not in masked.lower()
    assert "[artifact]" not in masked.lower()
    assert "[url]" not in masked.lower()


def test_canonical_url_unwraps_wayback_target():
    archived = canonical_article_url(
        "https://web.archive.org/web/20200101000000/https://www.example.com/article?utm_source=x"
    )
    assert archived == "https://example.com/article"
