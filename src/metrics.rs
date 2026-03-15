use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

#[derive(Clone)]
pub struct RequestMetricsHandle {
    request_id: u64,
    registry: Arc<MetricsRegistry>,
}

pub struct MetricsRegistry {
    next_request_id: AtomicU64,
    active_requests: Mutex<HashSet<u64>>,
}

impl MetricsRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            next_request_id: AtomicU64::new(1),
            active_requests: Mutex::new(HashSet::new()),
        })
    }

    pub fn start_request(self: &Arc<Self>, _streaming: bool) -> RequestMetricsHandle {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let mut requests = self.active_requests.lock().expect("metrics mutex poisoned");
        requests.insert(request_id);

        RequestMetricsHandle {
            request_id,
            registry: Arc::clone(self),
        }
    }

    fn finish_request(&self, request_id: u64) {
        let mut requests = self.active_requests.lock().expect("metrics mutex poisoned");
        requests.remove(&request_id);
    }

    fn fail_request(&self, request_id: u64) {
        let mut requests = self.active_requests.lock().expect("metrics mutex poisoned");
        requests.remove(&request_id);
    }
}

impl RequestMetricsHandle {
    pub fn id(&self) -> u64 {
        self.request_id
    }

    pub fn finish(&self, _input_tokens: u64, _output_tokens: u64) {
        self.registry.finish_request(self.request_id);
    }

    pub fn fail(&self) {
        self.registry.fail_request(self.request_id);
    }
}

impl Drop for RequestMetricsHandle {
    fn drop(&mut self) {
        self.registry.fail_request(self.request_id);
    }
}
