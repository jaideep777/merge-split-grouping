N = 1000

ws = rnorm(n = N, mean = 3, sd = 1) #numeric(N)*0
#wss = sort.int(ws, index.return = T)
#ws[(N/2):N] = 3
ws[ws < 0] = 0
wc = numeric(N)*0
gid = rep(0,N)

cs = 2
c  = 0.1
b = 100

ng = rep(0,N)
attractors = rep(0,N)
# make groups by distributing individuals randomly
gid = sample(x = 1:N, size = N, replace = T)  
g2ng_map = table(gid)
sites = as.integer(rownames(a))
ng[sites] = as.numeric(a) 
# disperse
for (i in 1:N){
  att_norm = attractors/(ng+1e-6)
  gid[i] = g
  attractors[g] = attractors[g]+ws[i]
  ng[g] = ng[g] + 1
}
#, prob = 0+1/(1+((att_norm-ws[i])/.1)^2)
hist(ng[ng>0])
plot(ng[ng>0]~att_norm[ng>0])
abline(lm(ng[ng>0]~att_norm[ng>0]))


# plot((dnorm((-100:100)/30)+rnorm(201,0,0))~seq(-100,100))
# abline(lm((dnorm((-100:100)/30)+rnorm(201,0,.1))~seq(-100,100)))
