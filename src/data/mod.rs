pub mod sources;
pub mod file;
pub mod collection;

pub use file::{DataFile, DataType, RedundancyLevel};
pub use sources::{DataSource, GitHubDataSource, StackOverflowDataSource, WikipediaDataSource};
pub use collection::{DataCollector, CollectionConfig, CollectionResults};