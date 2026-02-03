//! RAG configuration

use serde::{Deserialize, Serialize};

/// Configuration for RAG/pgvector connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// PostgreSQL connection string
    /// Format: postgres://user:password@host:port/database
    pub connection_string: String,
    
    /// Name of the embeddings table
    #[serde(default = "default_table_name")]
    pub table_name: String,
    
    /// Embedding dimension (must match your embedding model)
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
    
    /// Maximum number of results to return from search
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    
    /// Minimum similarity score (0.0 - 1.0) for results
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
    
    /// Connection pool size
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
    
    /// Distance metric for similarity search
    #[serde(default)]
    pub distance_metric: DistanceMetric,
}

fn default_table_name() -> String {
    "embeddings".to_string()
}

fn default_embedding_dim() -> usize {
    384 // Common for small embedding models like all-MiniLM-L6-v2
}

fn default_max_results() -> usize {
    5
}

fn default_min_similarity() -> f32 {
    0.5
}

fn default_pool_size() -> usize {
    10
}

/// Distance metric for vector similarity search
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    /// Cosine similarity (default, best for normalized embeddings)
    #[default]
    Cosine,
    /// L2 (Euclidean) distance
    L2,
    /// Inner product
    InnerProduct,
}

impl DistanceMetric {
    /// Get the pgvector operator for this metric
    pub fn operator(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "<=>",
            DistanceMetric::L2 => "<->",
            DistanceMetric::InnerProduct => "<#>",
        }
    }
    
    /// Get the index operator class for this metric
    pub fn index_ops(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "vector_cosine_ops",
            DistanceMetric::L2 => "vector_l2_ops",
            DistanceMetric::InnerProduct => "vector_ip_ops",
        }
    }
}

impl RagConfig {
    /// Create a new RAG configuration with just a connection string
    pub fn new(connection_string: impl Into<String>) -> Self {
        Self {
            connection_string: connection_string.into(),
            table_name: default_table_name(),
            embedding_dim: default_embedding_dim(),
            max_results: default_max_results(),
            min_similarity: default_min_similarity(),
            pool_size: default_pool_size(),
            distance_metric: DistanceMetric::default(),
        }
    }
    
    /// Set the embeddings table name
    pub fn with_table(mut self, table_name: impl Into<String>) -> Self {
        self.table_name = table_name.into();
        self
    }
    
    /// Set the embedding dimension
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }
    
    /// Set the maximum number of search results
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }
    
    /// Set the minimum similarity threshold
    pub fn with_min_similarity(mut self, min: f32) -> Self {
        self.min_similarity = min.clamp(0.0, 1.0);
        self
    }
    
    /// Set the distance metric
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }
    
    /// Set the connection pool size
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, super::RagError> {
        let connection_string = std::env::var("RAG_DATABASE_URL")
            .or_else(|_| std::env::var("DATABASE_URL"))
            .map_err(|_| super::RagError::ConfigError(
                "RAG_DATABASE_URL or DATABASE_URL environment variable not set".into()
            ))?;
        
        let mut config = Self::new(connection_string);
        
        if let Ok(table) = std::env::var("RAG_TABLE_NAME") {
            config.table_name = table;
        }
        
        if let Ok(dim) = std::env::var("RAG_EMBEDDING_DIM") {
            config.embedding_dim = dim.parse().map_err(|_| 
                super::RagError::ConfigError("Invalid RAG_EMBEDDING_DIM".into()))?;
        }
        
        if let Ok(max) = std::env::var("RAG_MAX_RESULTS") {
            config.max_results = max.parse().map_err(|_|
                super::RagError::ConfigError("Invalid RAG_MAX_RESULTS".into()))?;
        }
        
        if let Ok(min) = std::env::var("RAG_MIN_SIMILARITY") {
            config.min_similarity = min.parse().map_err(|_|
                super::RagError::ConfigError("Invalid RAG_MIN_SIMILARITY".into()))?;
        }
        
        Ok(config)
    }
}
