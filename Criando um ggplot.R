library(tidyverse)

ggplot2::mpg
#para abrir o data frame (o mpg contém observações coletadas pela Agência de Proteção Ambiental dos Estaddos Unidos sobre 38 modelos de carros)

ggplot(data=mpg)+geom_point(mapping=aes(x=displ,y=hwy))
#Criando um ggplot, com gráfico de mpg, onde displ no eixo x e hwy no y.

ggplot(data=mpg)+geom_point(mapping=aes(x=displ,y=hwy,color=class))
#Separado por cores

ggplot(data=mpg)+geom_point(mapping=aes(x=displ,y=hwy,size=class))
#Separado por tamanho,
#Recebe warning pois não é uma boa ideia mapear uma variável não ordenada (class) à uma estética ordenada (size)

ggplot(data=mpg)+geom_point(mapping=aes(x=displ,y=hwy,alpha=class))
#Separado por transparência dos pontos

ggplot(data=mpg)+geom_point(mapping=aes(x=displ,y=hwy,shape=class))
#Separado por formato dos pontos

ggplot(data=mpg)+geom_point(mapping=aes(x=displ,y=hwy),color="blue")
#Para mudar a cor

ggplot(data=mpg)+geom_smooth(mapping=aes(x=displ,y=hwy))
#Gráfico Geom de Ponto

ggplot(data=mpg)+geom_smooth(mapping=aes(x=displ,y=hwy,linetype=drv))
#Para configurar diferentes formas de linhas

ggplot(data=mpg)+geom_smooth(mapping=aes(x=displ,y=hwy,group=drv))
#Gráfico Geom Smooth

ggplot(data=mpg)+geom_smooth(mapping=aes(x=displ,y=hwy,color=drv),show.legend=FALSE)
#Para mudar a cor das linhas

ggplot(data=mpg)+geom_point(mapping=aes(x=displ,y=hwy))+geom_smooth(mapping=aes(x=displ,y=hwy))
#Para exibir vários Geoms no mmesmo gráfico

ggplot(data=mpg,mapping=aes(x=displ,y=hwy))+geom_point(mapping=aes(color=class))+geom_smooth()
#Geoms diferentes com camadas estéticas diferentes

ggplot(data=mpg,mapping=aes(x=displ,y=hwy))+geom_point(mapping=aes(color=class))+geom_smooth(data=filter(mpg,class=="subcompact"),se=FALSE)
#Para exibir apenas um subconjunto no gráfico, aqui utilizamos "subcompact" como e.g.



