use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

/// Dead-simple compression benchmark: Will Reducto Mode 3 save me storage/bandwidth costs on MY data?
#[derive(Parser, Debug)]
#[command(name = "benchmark")]
#[command(about = "A dead-simple benchmark that answers: Will Reducto Mode 3 save me storage/bandwidth costs on MY data?")]
#[command(version)]
struct Args {
    /// Path to directory containing your data files (optional - generates test data if not provided)
    #[arg(help = "Path to your data directory (max 100MB analyzed)")]
    data_path: Option<PathBuf>,
    
    /// Maximum time to run benchmark in seconds
    #[arg(long, default_value = "60")]
    timeout: u64,
    
    /// Output file for detailed results
    #[arg(long, default_value = "benchmark_results.txt")]
    output: PathBuf,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    if args.verbose {
        println!("Starting compression benchmark...");
        if let Some(ref path) = args.data_path {
            println!("Data path: {}", path.display());
        } else {
            println!("No data path provided - will generate 20MB test data");
        }
        println!("Timeout: {} seconds", args.timeout);
        println!("Output file: {}", args.output.display());
    }
    
// Enforce global 60-second timeout
    let fut = compression_benchmark_suite::run_benchmark(
        args.data_path,
        args.timeout,
        args.output.clone(),
        args.verbose,
    );
    let result = match tokio::time::timeout(std::time::Duration::from_secs(args.timeout), fut).await {
        Ok(Ok(r)) => Ok(r),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(compression_benchmark_suite::BenchmarkError::Timeout(args.timeout)),
    }?;
    let result = compression_benchmark_suite::run_benchmark(
        args.data_path,
        args.timeout,
        args.output,
        args.verbose,
    ).await?;
    
    // Print the final recommendation
    println!("\n{}", result.recommendation);
    
    if result.is_recommended {
        println!("Next steps: Add Reducto to your project with: cargo add reducto");
    }
    
    Ok(())
}