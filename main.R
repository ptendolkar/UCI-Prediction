library(R.matlab)
library(caret)
library(signal)
library(parallel)
library(doParallel)
library(reshape2)

set.seed(1234)

url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/00313/sEMG_Basic_Hand_movements_upatras.zip";

if(!file.exists("data.zip")){
  print("Downloading...")
  download.file(url, "data.zip")
}else{
  print("file exists.")
}

unzip("data.zip")


classify <- function(x){
  print(paste(x, "..."))
  dm <- createDesignMatrix(x)
  
  trainIndex <- createDataPartition(dm$class, p = 0.66, times=15)
  
  acc <- sapply(trainIndex, trainmod, dm=dm)
}sum

trainmod <- function(x, dm){
  training = dm[x, ]
  testing = dm[-x,]
  
  svmGrid <- data.frame(C     = c(rep(0.01,4), rep(0.1, 4), rep(1,4), rep(10,4), rep(100,4)),
                        sigma = rep(c(0.01, 0.1, 1, 10), 5) )
  
  knnGrid <- data.frame(k = c(1,3,5,7,9))
  
  gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                          n.trees = (1:30)*50,
                          shrinkage = 0.1)
  
  fitControl <- trainControl(## 10-fold CV
    method = "repeatedcv",
    number = 5)
  
  
  print("training...")  
  #run model in parallel
  cl <- makeCluster(detectCores())
  registerDoParallel(cl)
  
  svmFit <- train(class ~ ., data=training, method = "svmRadial", trControl = fitControl, tuneGrid = svmGrid)
  knnFit <- train(class ~ ., data=training, method = "knn", trControl = fitControl, tuneGrid = knnGrid)
  nbFit <- train(class ~ ., data=training, method = "nb", trControl = fitControl)
  gbmFit <- train(class ~ ., data = training, method = "gbm", trControl = fitControl, tuneGrid = gbmGrid, verbose = FALSE)
  
  stopCluster(cl)
  
  print("done")
  preds.svm <- predict(svmFit, newdata=testing)
  preds.knn <- predict(knnFit, newdata=testing)
  preds.nb <- predict(nbFit, newdata=testing)
  preds.gbm <- predict(gbmFit,newdata=testing)
  
  cf.svm <- confusionMatrix(preds.svm, testing$class)
  cf.knn <- confusionMatrix(preds.knn, testing$class)
  cf.nb <- confusionMatrix(preds.nb, testing$class)
  cf.gbm <- confusionMatrix(preds.gbm, testing$class)
  
  d<-rbind(cf.svm$overall[1], cf.knn$overall[1], cf.nb$overall[1], cf.gbm$overall[1])
  #rownames(d)<-c("svm", "knn", "nb", "gbm")
  d
}

createDesignMatrix <- function (x){
  fn <- readMat(paste(".\\Database 1\\", x, sep=""))
  nfn<- list(names(fn))
  
  fs <- 500
  fc <- 10
  flt <- butter(4, fc/(fs/2), type = "low")
  
  for (x in names(fn)){
    chunk <- fn[[x]]
    chunk2 <- matrix(nrow=nrow(chunk), ncol = ceiling(ncol(chunk) * fc*2 / fs))
    
    for(i in 1:nrow(chunk)){
      sig<-chunk[i,]
      sig<- abs(sig)  
      
      sig<- filtfilt(flt, sig)
      
      sig <- resample(sig, p=fc*2, q = fs)
      chunk2[i,]<-sig
    } 
    nfn[[x]] <- chunk2
  }
  fn<-nfn
  cyl <- cbind(fn$cyl.ch1, fn$cyl.ch2)
  hook <- cbind(fn$hook.ch1, fn$hook.ch2)
  tip <- cbind(fn$tip.ch1, fn$tip.ch2)
  palm <- cbind(fn$palm.ch1, fn$palm.ch2)
  sphere <- cbind(fn$spher.ch1, fn$spher.ch2)
  lat <- cbind(fn$lat.ch1, fn$lat.ch2)
  
  datamat <- rbind(cyl, hook, tip, palm, sphere, lat)
  
  datamat <- as.data.frame(datamat)
  
  datamat$class <- as.vector(cbind(rep("cyl", 30),
                                   rep("hook",30),
                                   rep("tip", 30),
                                   rep("palm",30),
                                   rep("sphere",30),
                                   rep("lat", 30)))
  
  datamat$class <- as.factor(datamat$class)
  
  
  datamat
}

subjects <- dir("Database 1")

a<-lapply(subjects, classify)

names(a) <- subjects
for(x in 1:length(a)){
  rownames(a[[x]]) <- c("svm", "knn", "nb", "gbm")
}

lapply(names(a), function(x){
  strtitle <- x
  x <- a[[x]]
  df <- melt(x, varnames = c("model", "sample"))
  
  fname <- paste(substring(strtitle, 1, nchar(strtitle)-4), ".png", sep="")
  print(fname)
  png(filename = fname, width=600, height = 600)
  #bwplot(value ~ model, data=df, main=strtitle, ylim=c(.6,1), ylab="Accuracy", xlab="Model")
  g <- ggplot(df, aes(model, value, color=model)) + geom_boxplot() + geom_point() 
  g <- g + xlab("Model")+ylab("Accuracy") +scale_y_continuous(limits=c(0.6, 1)) + geom_jitter()
  g <- g + ggtitle(substring(strtitle, 1, nchar(strtitle)-4))
  print(g)
  dev.off()
})


b<- lapply(names(a), function(x){
  x <- a[[x]]
  df <- melt(x, varnames = c("model", "sample"))
  df
})
names(b) <- names(a)
c <- melt(b)

library(lattice)
png("accuracylatticebwplot.png", width=1200, height=450,bg = "transparent")
bwplot(value ~ model | L1, c, layout=c(5,1), ylab="Accuracy", xlab="Model", ylim = c(0.65, 1), main="")
dev.off()
