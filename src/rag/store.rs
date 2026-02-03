//! RAG vector store with pgvector

use deadpool_postgres::{Config, Pool, Runtime};
use pgvector::Vector;
use tokio_postgres::NoTls;
use serde::{Deserialize, Serialize};

use super::{RagConfig, RagError, RagResult};

/// A document with its embedding stored in the vector database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier
    pub id: i64,
    /// Text content
    pub content: String,
    /// Optional metadata as JSON
    pub metadata: Option<serde_json::Value>,
    /// Similarity score from search (only populated in search results)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// A document to be inserted (without ID)
#[derive(Debug, Clone)]
pub struct NewDocument {
    /// Text content
    pub content: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// RAG vector store backed by PostgreSQL + pgvector
pub struct RagStore {
    pool: Pool,
    config: RagConfig,
}

impl RagStore {
    /// Connect to the vector store
    pub async fn connect(config: RagConfig) -> RagResult<Self> {
        let mut pg_config = Config::new();
        
        // Parse connection string
        let url = url::Url::parse(&config.connection_string)
            .map_err(|e| RagError::ConfigError(format!("Invalid connection string: {}", e)))?;
        
        pg_config.host = url.host_str().map(String::from);
        pg_config.port = url.port();
        pg_config.user = if url.username().is_empty() { None } else { Some(url.username().to_string()) };
        pg_config.password = url.password().map(String::from);
        pg_config.dbname = Some(url.path().trim_start_matches('/').to_string());
        
        let pool = pg_config
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        // Test connection
        let client = pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        // Verify pgvector extension is available
        client.query_one("SELECT extversion FROM pg_extension WHERE extname = 'vector'", &[])
            .await
            .map_err(|_| RagError::ConnectionFailed(
                "pgvector extension not installed. Run: CREATE EXTENSION vector;".into()
            ))?;
        
        Ok(Self { pool, config })
    }
    
    /// Create the embeddings table if it doesn't exist
    pub async fn create_table(&self) -> RagResult<()> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let create_table = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({}) NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            "#,
            self.config.table_name,
            self.config.embedding_dim
        );
        
        client.execute(&create_table, &[]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        // Create index for fast similarity search
        let create_index = format!(
            r#"
            CREATE INDEX IF NOT EXISTS {}_embedding_idx 
            ON {} USING ivfflat (embedding {})
            "#,
            self.config.table_name,
            self.config.table_name,
            self.config.distance_metric.index_ops()
        );
        
        // Index creation may fail if table is empty, that's okay
        let _ = client.execute(&create_index, &[]).await;
        
        Ok(())
    }
    
    /// Insert a document with its embedding
    pub async fn insert(&self, doc: NewDocument) -> RagResult<i64> {
        if doc.embedding.len() != self.config.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: doc.embedding.len(),
            });
        }
        
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let embedding = Vector::from(doc.embedding);
        
        let query = format!(
            "INSERT INTO {} (content, embedding, metadata) VALUES ($1, $2, $3) RETURNING id",
            self.config.table_name
        );
        
        let row = client.query_one(&query, &[&doc.content, &embedding, &doc.metadata]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(row.get(0))
    }
    
    /// Insert multiple documents in a batch
    pub async fn insert_batch(&self, docs: Vec<NewDocument>) -> RagResult<Vec<i64>> {
        let mut ids = Vec::with_capacity(docs.len());
        
        // TODO: Use COPY for better performance with large batches
        for doc in docs {
            let id = self.insert(doc).await?;
            ids.push(id);
        }
        
        Ok(ids)
    }
    
    /// Search for similar documents using vector similarity
    pub async fn search(&self, query_embedding: &[f32], limit: Option<usize>) -> RagResult<Vec<Document>> {
        if query_embedding.len() != self.config.embedding_dim {
            return Err(RagError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: query_embedding.len(),
            });
        }
        
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let embedding = Vector::from(query_embedding.to_vec());
        let limit = limit.unwrap_or(self.config.max_results) as i64;
        let operator = self.config.distance_metric.operator();
        
        // For cosine distance, convert to similarity (1 - distance)
        let score_expr = match self.config.distance_metric {
            super::DistanceMetric::Cosine => format!("1 - (embedding {} $1)", operator),
            super::DistanceMetric::L2 => format!("1 / (1 + (embedding {} $1))", operator),
            super::DistanceMetric::InnerProduct => format!("-(embedding {} $1)", operator),
        };
        
        let query = format!(
            r#"
            SELECT id, content, metadata, {} as score
            FROM {}
            WHERE {} >= $2
            ORDER BY embedding {} $1
            LIMIT $3
            "#,
            score_expr,
            self.config.table_name,
            score_expr,
            operator
        );
        
        let rows = client.query(&query, &[&embedding, &self.config.min_similarity, &limit]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        let docs = rows.iter().map(|row| {
            Document {
                id: row.get(0),
                content: row.get(1),
                metadata: row.get(2),
                score: Some(row.get(3)),
            }
        }).collect();
        
        Ok(docs)
    }
    
    /// Get a document by ID
    pub async fn get(&self, id: i64) -> RagResult<Option<Document>> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let query = format!(
            "SELECT id, content, metadata FROM {} WHERE id = $1",
            self.config.table_name
        );
        
        let row = client.query_opt(&query, &[&id]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(row.map(|r| Document {
            id: r.get(0),
            content: r.get(1),
            metadata: r.get(2),
            score: None,
        }))
    }
    
    /// Delete a document by ID
    pub async fn delete(&self, id: i64) -> RagResult<bool> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let query = format!("DELETE FROM {} WHERE id = $1", self.config.table_name);
        let affected = client.execute(&query, &[&id]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(affected > 0)
    }
    
    /// Count total documents
    pub async fn count(&self) -> RagResult<i64> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let query = format!("SELECT COUNT(*) FROM {}", self.config.table_name);
        let row = client.query_one(&query, &[]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(row.get(0))
    }
    
    /// Clear all documents from the table
    pub async fn clear(&self) -> RagResult<u64> {
        let client = self.pool.get().await
            .map_err(|e| RagError::ConnectionFailed(format!("{}", e)))?;
        
        let query = format!("DELETE FROM {}", self.config.table_name);
        let affected = client.execute(&query, &[]).await
            .map_err(|e| RagError::QueryFailed(format!("{}", e)))?;
        
        Ok(affected)
    }
    
    /// Get the configuration
    pub fn config(&self) -> &RagConfig {
        &self.config
    }
}

/// Builder for creating RAG context from search results
pub struct RagContextBuilder {
    docs: Vec<Document>,
    separator: String,
    max_tokens: Option<usize>,
    include_scores: bool,
}

impl RagContextBuilder {
    /// Create a new context builder from search results
    pub fn new(docs: Vec<Document>) -> Self {
        Self {
            docs,
            separator: "\n\n".to_string(),
            max_tokens: None,
            include_scores: false,
        }
    }
    
    /// Set the separator between documents
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }
    
    /// Set approximate maximum tokens (characters / 4)
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }
    
    /// Include similarity scores in output
    pub fn with_scores(mut self, include: bool) -> Self {
        self.include_scores = include;
        self
    }
    
    /// Build the context string
    pub fn build(self) -> String {
        let mut parts = Vec::new();
        let mut total_chars = 0;
        let max_chars = self.max_tokens.map(|t| t * 4);
        
        for doc in &self.docs {
            let part = if self.include_scores {
                if let Some(score) = doc.score {
                    format!("[{:.2}] {}", score, doc.content)
                } else {
                    doc.content.clone()
                }
            } else {
                doc.content.clone()
            };
            
            if let Some(max) = max_chars {
                if total_chars + part.len() > max {
                    break;
                }
            }
            
            total_chars += part.len() + self.separator.len();
            parts.push(part);
        }
        
        parts.join(&self.separator)
    }
    
    /// Build a prompt with context and question
    pub fn build_prompt(self, question: &str) -> String {
        let context = self.build();
        format!(
            "Use the following context to answer the question.\n\n\
            Context:\n{}\n\n\
            Question: {}\n\n\
            Answer:",
            context,
            question
        )
    }
}
