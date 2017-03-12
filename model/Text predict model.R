#This code supports the app deployed on shinyapp called "wordpredict", part of the Coursera Specialization 


# begin with cleaning env and load libraries, set dir
rm(list=ls())

library(dplyr); library(stringr); library(RWeka)  ; library(ngram); library(quanteda); 
library(knitr); library(xtable) ; library(textreg); library(caret);

setwd("C:/Users/Alberto/Documents/Datascience Coursera/08. Capstone Project/app/wordpredict/data") #app/wordpredict/

# connect files 
USblog    <- file("./final/en_US/en_US.blogs.txt", "r") 
UStwitter <- file("./final/en_US/en_US.twitter.txt", "r")
USnews    <- file("./final/en_US/en_US.news.txt", "r")

# read in the text files provided
readblog    <- readLines(USblog, encoding = "UTF-8", skipNul=TRUE)
readtwitter <- readLines(UStwitter, encoding = "UTF-8", skipNul=TRUE)
readnews    <- readLines(USnews, encoding = "UTF-8", skipNul=TRUE)
close(USblog); close(USnews); close(UStwitter)

# attach attribute comment to be used when writing different files to .csv
comment(readblog)     <- "blog"
comment(readtwitter)  <- "twitter"
comment(readnews)     <- "news"

# function to select partitioning to apply
partitioning   <- function (x, percent = 0.2) {
                      # input are text file and % you want to sample, output are test and traing partitions of your text
                      filename <- attributes(x)$comment
                      size <- round(length(x)*percent)
                      set.seed(123)
                      v <- sample(1:length(x), size)                  
                      temp_train <- x[v]
                      temp_test <- x[-v]
                      write(temp_train, file = paste0("train_", filename,"_", percent,".txt"))
                      write(temp_test, file = paste0("test_", filename, "_", 1-percent,".txt"))
    }

partitioning(readblog, 0.3)
partitioning(readtwitter, 0.3)
partitioning(readnews, 0.3)

rm(readblog, readnews, readtwitter, USblog, USnews, UStwitter)

# read each train file in R
trainblog    <- readLines("./train_blog_0.3.txt", encoding = "UTF-8", skipNul=TRUE)
traintwitter    <- readLines("./train_twitter_0.3.txt", encoding = "UTF-8", skipNul=TRUE)
trainnews    <- readLines("./train_news_0.3.txt", encoding = "UTF-8", skipNul=TRUE)

# clean text 
clean_text <- function (x) {
  
                    # clean
                    x <- gsub("can't", "cann't", x) # otherwise next steps generates ca not
                    x <- gsub("n't", " not", x) # abbreviations of not to regular
                    x <- gsub(" www(.+) ", "", x) # remove website addresses
                    x <- gsub("@\\w+ *", "", x) # remove word starting with @
                    x <- gsub("#\\w+ *", "", x) # remove word starting with #
                    x <- gsub("[^[:alpha:] ]", "", x) # keep all except alphabetic chars
                    
                    # whitespace optimizing at end after punctuation is removed
                    x <- gsub("\\s+", " ", x)   # reduce one or more whitespace to just one
                    x <- gsub(" $","", x)       # remove whitespace end of the sentences
                    x <- tolower(x)
                    
                    return (x)
}

#apply cleaning function to three texts
clean_sampleblog    <- clean_text (trainblog)
clean_sampletwitter <- clean_text (traintwitter)
clean_samplenews    <- clean_text (trainnews)
rm(trainblog, traintwitter, trainnews)

comment(clean_sampleblog) <- "blog"
comment(clean_sampletwitter) <- "twitter"
comment(clean_samplenews) <- "news"

# Tokenize using Quanteda, create freq tables and save to file
tokenized_df <- function (x, n) {
                
                    filename <- attributes(x)$comment
                    temp <- tokenize(x, what = "word", ngrams = n, concatenator = " ", simplify = TRUE, verbose = TRUE)
                    
                    # Create freq table
                    temp <- tbl_df(temp)
                    colnames(temp) <- "gram"
                    temp <- group_by(temp, gram)%>%tally()%>%arrange(desc(n))
                    
                    # write to file
                    write.csv(temp, file = paste0(n, "gram", filename, ".csv"), row.names=FALSE)
                    rm(temp)
}

# generate 2 grams database
tokenized_df(clean_samplenews, 2); tokenized_df(clean_sampletwitter, 2); tokenized_df(clean_sampleblog, 2)

twogram_blog    <- read.csv("./2gramblog.csv"   , header = TRUE, stringsAsFactors = FALSE)
twogram_twitter <- read.csv("./2gramtwitter.csv", header = TRUE, stringsAsFactors = FALSE)
twogram_news    <- read.csv("./2gramnews.csv"   , header = TRUE, stringsAsFactors = FALSE)

twogram_all <- tbl_df(rbind(twogram_blog, twogram_news, twogram_twitter))

##remove singletones from 2-grams (in the csv files loaded on shiny we removed below 5 of frequency)
twogram_all <- group_by(twogram_all, gram) %>% summarize(freq = sum(n)) %>% arrange(desc(freq))%>%filter(freq>1)
write.csv(twogram_all, file = "twogramall.csv", row.names=FALSE)
rm(twogram_all, twogram_blog, twogram_news, twogram_twitter)

# generate 3 grams database
tokenized_df(clean_samplenews, 3); tokenized_df(clean_sampletwitter, 3); tokenized_df(clean_sampleblog, 3)

threegram_blog    <- read.csv("./3gramblog.csv"   , header = TRUE, stringsAsFactors = FALSE)
threegram_twitter <- read.csv("./3gramtwitter.csv", header = TRUE, stringsAsFactors = FALSE)
threegram_news    <- read.csv("./3gramnews.csv"   , header = TRUE, stringsAsFactors = FALSE)

threegram_all <- tbl_df(rbind(threegram_blog, threegram_news, threegram_twitter))

threegram_all <- group_by(threegram_all, gram) %>% summarize(freq = sum(n)) %>% arrange(desc(freq))%>%filter(freq>1)
write.csv(threegram_all, file = "threegramall.csv", row.names=FALSE)
rm(threegram_all, threegram_blog, threegram_news, threegram_twitter)

# generate and store / read 4 grams database
tokenized_df(clean_samplenews, 4); tokenized_df(clean_sampletwitter, 4); tokenized_df(clean_sampleblog, 4)

fourgram_blog    <- read.csv("./4gramblog.csv"   , header = TRUE, stringsAsFactors = FALSE)
fourgram_twitter <- read.csv("./4gramtwitter.csv", header = TRUE, stringsAsFactors = FALSE)
fourgram_news    <- read.csv("./4gramnews.csv"   , header = TRUE, stringsAsFactors = FALSE)

fourgram_all <- tbl_df(rbind(fourgram_blog, fourgram_news, fourgram_twitter))

fourgram_all <- group_by(fourgram_all, gram) %>% summarize(freq = sum(n)) %>% arrange(desc(freq))%>%filter(freq>1)
write.csv(fourgram_all, file = "fourgramall.csv", row.names=FALSE)
rm(fourgram_all, fourgram_blog, fourgram_news, fourgram_twitter)


#READ FILES ALREADY PROCESSED##
df2 <- read.csv("twogramall.csv", stringsAsFactors = FALSE, sep = ",", header = TRUE);df3 <- read.csv("threegramall.csv", stringsAsFactors = FALSE);df4 <- read.csv("fourgramall.csv", stringsAsFactors = FALSE)

# algorithm with the following logic: "linear interpolation": cut the input string in pieces and feed to match all the ngrams databases, assign a lambda the collect the results in one table 
linear_interpolation <- function (x) {
  
              # remove nonalphabetic char + lower case + remove white space at input     
              string <- gsub("[^[:alpha:] ]", "", x)
              string <- tolower(string)
              string <- str_trim(string, side = "both")
              
              # keep the last 1 or 2 words of the string
              str_len <- wordcount(string, sep = " ")
              string3 <- word(string, str_len-2, -1)
              string2 <- word(string, str_len-1, -1)
              string1 <- word(string, -1)
              
              # also parametrize the lamba depending on input string
              # create a table with the lambdas
              length1 <- c(NA, NA, 1)
              length2 <- c(NA, 0.7, 0.3)
              length3 <- c(0.5, 0.3, 0.2)
              lambdas <- data.frame(length1, length2, length3)
              
              # if else to variate lambda give input length (1, 2 or >3)
              ifelse(str_len == 1 | str_len == 2, lambda_len <- str_len, lambda_len <- 3)
                
              # look up in our dataframes
              table3 <- dplyr::filter(df4, str_detect(gram, paste0("^", string3," "))) %>% 
                        dplyr::mutate(pred = word(gram, -1), p = round(freq/sum(freq),4)*(lambdas[1, ][[lambda_len]])) %>% 
                        select(pred, p) # keep only newly computed variables
                    
              table2 <- dplyr::filter(df3, str_detect(gram, paste0("^", string2," "))) %>% 
                        dplyr::mutate(pred = word(gram, -1), p = round(freq/sum(freq),4)*(lambdas[2, ][[lambda_len]])) %>% 
                        select(pred, p) # keep only newly computed variables
              
              table1 <- dplyr::filter(df2, str_detect(gram, paste0("^", string1," "))) %>% 
                        dplyr::mutate(pred = word(gram, -1), p = round(freq/sum(freq),4)*(lambdas[3, ][[lambda_len]])) %>% 
                        select(pred, p) # keep only newly computed variables
              
              table <- rbind(table1, table2, table3) %>% 
                        group_by(pred) %>%
                        summarize(p = sum(p)) %>% 
                        arrange(desc(p)) %>% 
                        top_n(15) #top_n() will remove NAs automatically  

              if (nrow(table) == 0) {
               
                  print("Sorry out of ideas")
                  
                  } 
              
              # print first 15 results
              print(table, quote = FALSE)
}


