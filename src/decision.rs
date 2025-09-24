// Decision engine: compare Reducto vs gzip results and make recommendation
use crate::compression::CompressResult;

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Recommended,
    NotRecommended,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub decision: Decision,
    pub reason: String,
}

/// Apply rules:
/// 1. Reducto ratio must be greater than gzip ratio by at least 1% (to avoid noise)
/// 2. Reducto time must not be >10x slower than gzip
pub fn compare_results(reducto: &CompressResult, gzip: &CompressResult) -> Recommendation {
    let ratio_gain = reducto.ratio / gzip.ratio;
    let speed_penalty = reducto.time_ms as f64 / gzip.time_ms as f64;

    if ratio_gain > 1.01 && speed_penalty <= 10.0 {
        Recommendation {
            decision: Decision::Recommended,
            reason: format!(
                "Reducto ratio {:.2}x vs gzip {:.2}x ({}% better), speed penalty {:.1}x",
                reducto.ratio,
                gzip.ratio,
                ((ratio_gain - 1.0) * 100.0).round(),
                speed_penalty
            ),
        }
    } else {
        Recommendation {
            decision: Decision::NotRecommended,
            reason: if ratio_gain <= 1.01 {
                "Compression ratio improvement insufficient".into()
            } else {
                format!("Reducto is {:.1}x slower than gzip (limit 10x)", speed_penalty)
            },
        }
    }
}
