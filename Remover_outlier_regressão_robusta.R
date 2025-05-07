#Como remover um outlier através da Regressão linear robusta (Huber)

Library(MASS)

x<-1:19

y<-x-rnorm(19,0,1)

x[20]<-22

y[20]<-80

fit<-lm(y~x)

fit2<-rlm(y~x)

plot(x,y,lwd=3,xlim=c(0,25),ylim=c(0,85),cex=3)

abline(fit,lwd=4,col="green3")

abline(fit2,lwd=4,col="blue")

legend(0,85,legend=c("linear regression(OLS)","Robust linear regression(Huber)"),col=c("green3","blue"),lty=1,cex=1.5,lwd=4)
