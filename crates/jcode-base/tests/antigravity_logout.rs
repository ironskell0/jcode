use jcode_base::auth::antigravity::{self, AntigravityTokens};

struct RestoreJcodeHome(Option<std::ffi::OsString>);

impl Drop for RestoreJcodeHome {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            unsafe { std::env::set_var("JCODE_HOME", value) };
        } else {
            unsafe { std::env::remove_var("JCODE_HOME") };
        }
    }
}

#[test]
fn antigravity_logout_removes_saved_jcode_credentials() -> anyhow::Result<()> {
    let home = tempfile::tempdir()?;
    let _restore_home = RestoreJcodeHome(std::env::var_os("JCODE_HOME"));
    unsafe { std::env::set_var("JCODE_HOME", home.path()) };

    antigravity::save_tokens(&AntigravityTokens {
        access_token: "test-access".to_string(),
        refresh_token: "test-refresh".to_string(),
        expires_at: i64::MAX,
        email: None,
        project_id: None,
    })?;
    let path = antigravity::tokens_path()?;
    assert!(path.is_file());

    antigravity::clear_tokens()?;

    assert!(!path.exists());
    Ok(())
}
