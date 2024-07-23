# How to run from the command line: Rscript download_written_questions.R

# Uncomment these lines before execution if you don't have them installed yet
#require(devtools)
#install_github("PolscienceAntwerp/flempar")
#install.packages("dplyr")
library(flempar)
library(dplyr)

# Get all the written questions between "2024-01-01" and "2024-06-30" in a dataframe
wq_documents <- get_work(date_range_from="2024-01-01",
                         date_range_to="2024-06-30",
                         type="document",
                         fact="written_questions"
)

# write to csv file
write.csv(wq_documents, "../written_questions_202401_202406.csv", row.names=FALSE)