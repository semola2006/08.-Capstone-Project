# begin with cleaning env and load libraries, set dir
rm(list=ls())

library(dplyr); library(stringr); library(RWeka)  ; library(ngram); library(quanteda); 
library(knitr); library(xtable) ; library(textreg); library(caret);

setwd("C:/Users/Alberto/Documents/Datascience Coursera/08. Capstone Project/app/wordpredict/data") #app/wordpredict/

# connect files 
USblog    <- file("./final/en_US/en_US.blogs.txt", "r") 
UStwitter <- file("./final/en_US/en_US.twitter.txt", "r")
USnews    <- file("./final/en_US/en_US.news.txt", "r")

# read 
readblog    <- readLines(USblog, encoding = "UTF-8", skipNul=TRUE)
readtwitter <- readLines(UStwitter, encoding = "UTF-8", skipNul=TRUE)
readnews    <- readLines(USnews, encoding = "UTF-8", skipNul=TRUE)
close(USblog); close(USnews); close(UStwitter)

# attach attribute comment to be used when writeing differnet files
comment(readblog)     <- "blog"
comment(readtwitter)  <- "twitter"
comment(readnews)     <- "news"

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

#apply function to three texts
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

# generate 1 gram database
tokenized_df(clean_samplenews,1); tokenized_df(clean_sampletwitter,1); tokenized_df(clean_sampleblog,1)

onegram_blog    <- read.csv("./1gramblog.csv",    header = TRUE, stringsAsFactors = FALSE)
onegram_twitter <- read.csv("./1gramtwitter.csv", header = TRUE, stringsAsFactors = FALSE)
onegram_news    <- read.csv("./1gramnews.csv",    header = TRUE, stringsAsFactors = FALSE)

onegram_all <- tbl_df(rbind(onegram_blog, onegram_news, onegram_twitter))

#compile the freq table and remove frequencies below 5
onegram_all <- group_by(onegram_all, gram) %>% summarize(freq = sum(n)) %>% arrange(desc(freq))%>%filter(freq>=5)
write.csv(onegram_all, file = "onegramall.csv", row.names=FALSE)
rm(onegram_all, onegram_blog, onegram_news, onegram_twitter)

# generate 2 grams database
tokenized_df(clean_samplenews, 2); tokenized_df(clean_sampletwitter, 2); tokenized_df(clean_sampleblog, 2)

twogram_blog    <- read.csv("./2gramblog.csv"   , header = TRUE, stringsAsFactors = FALSE)
twogram_twitter <- read.csv("./2gramtwitter.csv", header = TRUE, stringsAsFactors = FALSE)
twogram_news    <- read.csv("./2gramnews.csv"   , header = TRUE, stringsAsFactors = FALSE)

twogram_all <- tbl_df(rbind(twogram_blog, twogram_news, twogram_twitter))

##remove singletones from 2-grams and 3-grams
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

# only to polish the ngram models saved to csv
df2$gram <- gsub("_", " ", df2$gram)
df3$gram <- gsub("_", " ", df3$gram)

write.csv(df3, file = "threegramall.csv", row.names=FALSE)

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


# function adjusted to fit shiny app
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
  table3a <-  dplyr::filter(df4, str_detect(df4$gram, paste0("^", string3," "))) 
  table3b <-  dplyr::mutate(table3a, pred = word(table3a$gram, -1), p = round(table3a$freq/sum(table3a$freq),4)*(lambdas[1, ][[lambda_len]])) 
  table3  <-  select(table3b, pred, p) # keep only newly computed variables
  
  table2a <- dplyr::filter(df3, str_detect(df3$gram, paste0("^", string2," "))) 
  table2b <- dplyr::mutate(table2a, pred = word(table2a$gram, -1), p = round(table2a$freq/sum(table2a$freq),4)*(lambdas[2, ][[lambda_len]])) 
  table2  <- select(table2b, pred, p) # keep only newly computed variables
  
  table1a <- dplyr::filter(df2, str_detect(df2$gram, paste0("^", string1," "))) 
  table1b <- dplyr::mutate(table1a, pred = word(table1a$gram, -1), p = round(table1a$freq/sum(table1a$freq),4)*(lambdas[3, ][[lambda_len]]))  
  table1  <- select(table1b, pred, p) # keep only newly computed variables
  
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

# function input - BASIC MODEL
input <- function (x) {
          
          # remove nonalphabetic char + lower case + remove white space at input     
          string <- gsub("[^[:alpha:] ]", "", x)
          string <- tolower(string)
          string <- str_trim(string, side = "both")
            
          #store length of string
          str_len <- wordcount(string, sep = " ")
          
          # set min, med and max string to lookup cases, max: more than 3 words min&med: to back off if first result NA
          max_string <- word(string, str_len-2, -1)
          med_string <- word(string, str_len-1, -1)
          min_string <- word(string, str_len  , -1)
          
          # test string length and grab based on match beginning of grams in dataframes
          if (str_len == 1) z <- as.data.frame(df2[grep(paste0("^", string, " "), df2$gram),])#$gram
          
          
          ###here untouched ->
          if (str_len == 2) z <- df3[grep(paste0("^", string, " "), df3$gram)[1:5], ]#$gram 
          
          # Need to handle case of outcome in NA - reduce string to match 2, 3 grams, otherwise sample of most frequent monogram
          if (z %in% NA) z <- df3[grep(paste0("^", med_string, " "), df3$gram)[1:5], ]$gram 
          if (z %in% NA) z <- df2[grep(paste0("^", min_string, " "), df2$gram)[1:5], ]$gram 
          if (z %in% NA) z <- sample(head(df1$gram))[1:5] 
          
          #trim right space and return only last word of string 
          outcome <- str_trim(z, side = "both")
          print(outcome, quote = FALSE)
}

model <- function (input) {
  
  # lookup into ngram model
  #   if found -> extract matching
  #     calculate probabilities for matching
  # 
  # if not found -> look up in ngram lower level 
  #     if found-> calcuate probabilities
  # 
  # if not found -> return a random?
  
  
}
