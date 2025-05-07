#Transformações Estatîsticas
ggplot(data=diamonds)+geom_bar(mapping=aes(x=cut))
#Gráfico de barras com informações sobre ~54.000 diamantes

#ou

ggplot(data=diamonds)+stat_count(mapping=aes(x=cut))
#Com o stat_acount consegue recriar o mesmo gráfico, pois o geom_bar(), usa do stat_count()

demo<-tribble(~a,~b,"bar_1",20,"bar_2",30,"bar_3",40)
#Para sobrescrever o stat padrão

ggplot(data=demo)+geom_bar(mapping=aes(x=a,y=b),stat="identity")
#Para aplicar ao gráfico a nova identidade, o novo stat padrão

ggplot(data=diamonds)+geom_bar(mapping=aes(x=cut,y=..prop..,group=1))
#Para exibir um gráfico de barras de proportion, em vez de count

ggplot(data=diamonds)+stat_summary(mapping=aes(x=cut,y=depth),fun.ymin=min,fun.ymax=max,fun.y=median)
#Para exibir mais de 20 stats para uso

ggplot(data=diamonds)+geom_bar(mapping=aes(x=cut,color=cut))
#Para colorir com a estética color

ggplot(data=diamonds)+geom_bar(mapping=aes(x=cut,fill=cut))
#Para Colorir com a estética fill

ggplot(data=diamonds)+geom_bar(mapping=aes(x=cut,fill=clarity))
#Para colorir de acordo com a claridade

#Caso não queira um gráfico de barras empilhadas, pode usar uma das três opções: "identity", "dodge" ou "fill":

#position="identity" ("alpha=1/5" ou "fill=NA")
ggplot(data=diamonds,mapping=aes(x=cut,fill=clarity))+geom_bar(alpha=1/5,position="identity")

#OU

ggplot(data=diamonds,mapping=aes(x=cut,color=clarity))+geom_bar(fill=NA,position="identity")


#position="fill" (empilha cada grupo de barras na mesma altura)
ggplot(data=diamonds)+geom_bar(mapping=aes(x=cut,fill=clarity),position="fill")


#position="dodge"(coloca objetos sobrepostos diretamente um ao lado do outro, facilitando comparações individuais)
ggplot(data=diamonds)+geom_bar(mapping=aes(x=cut,fill=clarity),position="dodge")


ggplot(data=mpg)+geom_point(mapping=aes(x=displ,y=hwy),position="jitter")
#Para evitar overploting (pontos se sobrepondo), utiliza-se o position="jitter"
#Pois adiciona uma pequena quantidade de ruîdo aleatório a cada ponto, espalhando-os


