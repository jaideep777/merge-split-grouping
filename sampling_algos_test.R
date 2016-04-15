N = 512
id = 1:N
x = sample(x = seq(1,70, length.out=N),size = N, replace=T, prob=c(seq(0,1, length.out=0.7*N),seq(1,0, length.out=0.3*N)) )
hist(x)
Ki.sd = 5
prob = dnorm(x = x, mean=0, sd = Ki.sd)*sqrt(2*pi*Ki.sd^2)
prob2 = exp(-x*x/Ki.sd/Ki.sd)
plot(prob~x)

# Sample 1 - rejection sampling
n.samples = 10000
rs = numeric(n.samples)
iters.req = numeric(n.samples)
for (i in 1:n.samples){
  accept= F
  r = -1
  niter = 0
  while (accept == F){
    chosen.one = sample(id,1)
    if (runif(1) < prob[chosen.one]){
      r = chosen.one
      accept = T
    }
    niter = niter+1
  }
  rs[i] = r
  iters.req[i] = niter
  if (i %% 100 == 0) cat(i,"\n")
}


# roulett wheel sample
prob = c(5, 2, 0, 0, 0, 0, 10, 10, 0.01, 5)
N = length(prob)
cumm.prob = 0;
for(i in 1:length(prob)){
  cumm.prob = c(cumm.prob, cumm.prob[i]+prob[i])
}


for(i in 1:10000){
  a = runif(1)*cumm.prob[N+1]  # must be in [0,1)
  r = 1
  lo=1
  hi=N+1
  mid=as.integer((hi+lo)/2)
  while(hi != mid && lo != mid){
    if (cumm.prob[mid] > a){
      hi = mid
      mid=as.integer((hi+lo)/2)
    }
    else{
      lo = mid
      mid=as.integer((hi+lo)/2)
    }
  }
  r = lo
  while(cumm.prob[r] <= a){
   r=r+1
  }

  rs[i] = r-1
  if (i %% 100 == 0) cat(i,"\n")
}


# R internal sample
for(i in 1:10000){
  rs[i] = sample(id, size=1, prob=prob)
}

ids.freq = as.numeric(table(rs))
ids.prob = prob[as.integer(rownames(table(rs)))+1]
plot(ids.freq~ids.prob)
hist(iters.req)
hist(rs, 1000)
points(prob*700~seq(1,1000,1), col="red", type="o")

dat = read.delim("/home/jaideep/chapter3/reject_out.txt", header=F)
rs = dat$V1
table(dat$V1)/10000*sum(prob)

dat2 = read.delim("/home/jaideep/chapter3/roulette_setup.txt", header=F)
prob=dat2$V4
plot(prob~dat2$V3)

dat3 = read.delim("/home/jaideep/chapter3/pd.txt", header=F)
x = as.numeric(dat3[1,])
prob = exp(-x*x/5/5/2)
prob=x
plot(prob~x)
dat4 = read.delim("/home/jaideep/chapter3/reject_out_gpu.txt", header=F)
dat4 = dat4[-513]
rs = dat4$V1
iters = dat4$V2
hist(iters)
rs[rs > 512 | rs < 0] = NA

sum=0
for(tid in 1:512){
  prob = as.numeric(dat3[tid,])[-513]
  # prob = as.numeric(exp(-x*x/5/5/2))
  # plot(prob~x)
  ids = as.numeric(dat4[,tid])
  ids.freq = as.numeric(table(ids))
  ids.prob = prob[as.integer(rownames(table(ids)))+1]
  png(filename = sprintf("/home/jaideep/chapter3/figs_reject/%g.png", tid))
  plot(ids.freq~ids.prob)
  dev.off()
  cat(tid, "\n")
  sum = sum + length(which(ids == (tid-1)))
}

