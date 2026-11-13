from core.storage import verify_admin_password


def test_verify_admin_password_fail_closed_and_exact_match():
    assert verify_admin_password("coach-secret", "coach-secret") is True
    assert verify_admin_password("wrong", "coach-secret") is False
    assert verify_admin_password("", "") is False
    assert verify_admin_password("anything", None) is False
    assert verify_admin_password("", "coach-secret") is False
