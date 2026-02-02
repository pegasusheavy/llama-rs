//! Tensor data storage

use std::sync::Arc;

/// Owned or borrowed tensor data
#[derive(Debug, Clone)]
pub enum TensorStorage {
    /// Owned data on CPU
    Owned(Arc<Vec<u8>>),
    /// View into external data (e.g., memory-mapped file)
    View { data: *const u8, len: usize },
}

// SAFETY: View data comes from memory-mapped files which are thread-safe for reads
unsafe impl Send for TensorStorage {}
unsafe impl Sync for TensorStorage {}

impl TensorStorage {
    pub fn owned(data: Vec<u8>) -> Self {
        Self::Owned(Arc::new(data))
    }

    /// # Safety
    /// The data pointer must be valid for the lifetime of this storage.
    pub unsafe fn view(data: *const u8, len: usize) -> Self {
        Self::View { data, len }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Owned(data) => data.as_slice(),
            Self::View { data, len } => unsafe { std::slice::from_raw_parts(*data, *len) },
        }
    }

    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::Owned(data) => Arc::get_mut(data).map(|v| v.as_mut_slice()),
            Self::View { .. } => None,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::View { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_owned(&self) -> Self {
        Self::Owned(Arc::new(self.as_bytes().to_vec()))
    }
}
