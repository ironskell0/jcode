use anyhow::Result;
pub use jcode_provider_vertex::{AVAILABLE_MODELS, DEFAULT_LOCATION, DEFAULT_MODEL, PersistedCatalog, is_known_model};

pub fn persisted_catalog_path() -> Result<std::path::PathBuf> {
    Ok(crate::storage::app_config_dir()?.join("vertex_models_cache.json"))
}

pub fn load_persisted_catalog() -> Option<PersistedCatalog> {
    let path = persisted_catalog_path().ok()?;
    crate::storage::read_json(&path)
        .ok()
        .filter(|catalog: &PersistedCatalog| !catalog.models.is_empty())
}

pub fn persist_catalog(catalog: &PersistedCatalog) {
    if catalog.models.is_empty() {
        return;
    }
    let Ok(path) = persisted_catalog_path() else {
        return;
    };
    let _ = crate::storage::write_json(&path, catalog);
}
