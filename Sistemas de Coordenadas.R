#Sistemas de Coordenadas
library(tidyverse)

ggplot2::mpg


ggplot(data=mpg,mapping=aes(x=class,y=hwy))+geom_boxplot()
#Diagramas de caixas

ggplot(data=mpg,mapping=aes(x=class,y=hwy))+geom_boxplot()+coord_flip()
#Para deixar o boxplot na horizontal



nz<-map_data("nz")
#Para abrir banco de dados

ggplot(nz,aes(long,lat,group=group))+geom_polygon(fill="white",color="black")
#Para gerar mapa

ggplot(nz,aes(long,lat,group=group))+geom_polygon(fill="white",color="black")+coord_quickmap()


bar<-ggplot(data=diamonds)+geom_bar(mapping=aes(x=cut,fill=cut),show.legend=FALSE,width=1)+theme(aspect.ratio=1)+labs(x=NULL,y=NULL)

bar+coord_flip()

bar+coord_polar()                                                                                                                      