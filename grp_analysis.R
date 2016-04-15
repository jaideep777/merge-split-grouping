dat = read.delim("/home/jaideep/chapter3/gsd.txt", header=F, sep=" ")
h=hist(as.numeric(dat))
plot(log(h$counts)~h$mids)
table(as.numeric(dat))
plot(log(as.numeric(t))~as.integer(rownames(t)) )

dat1 = read.delim("/home/jaideep/chapter3/ngrps.txt", header=F, sep=" ")
plot(dat1$V1, type="l")



