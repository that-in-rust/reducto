// Formatter for benchmark results
use crate::compression::CompressResult;
use crate::decision::{Recommendation, Decision};
use std::fs::File;
use std::io::Write;
use anyhow::Result;

pub fn format_console(reducto: &CompressResult, gzip: &CompressResult, rec: &Recommendation) -> String {
    format!(
        "Compression Benchmark Results\n================================\n\nReducto Mode 3:\n  Ratio: {:.2}x\n  Time : {} ms\n\nGzip (lvl6):\n  Ratio: {:.2}x\n  Time : {} ms\n\nRecommendation: {:?}\nReason: {}\n",
        reducto.ratio,
        reducto.time_ms,
        gzip.ratio,
        gzip.time_ms,
        rec.decision,
        rec.reason
    )
}

pub fn save_report(path: &std::path::Path, report: &str) -> Result<()> {
    let mut file = File::create(path)?;
    file.write_all(report.as_bytes())?;
    Ok(())
}
