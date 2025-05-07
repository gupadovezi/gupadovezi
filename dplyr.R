#Como utilizar o dplyr através de um conjunto de dados sobre voos partindo de NY city em 2013
install.packages("nycflights13")

library(nycflights13)

library(tidyverse)

nycflights13::flights

View(flights)
#utils

#ou

view(flights)
#tribble


#Para filtrar linhas com filter:
filter(flights,month==1,day==1)

jan1<-filter(flights,month==1,day==1)
#Para salvar resultados precisará do "<-"

(dec25<-filter(flights,month==12,day==25))
##Para imprimir os resultados e salvar em uma variável ao mesmo tempo


#Comparações (>,>=,<,<=,!=(diferente),e ==(igual)):

filter(flights,month=1)
#Error in `filter()`:
#! We detected a named input.
#ℹ This usually means that you've used `=` instead of `==`.
#ℹ Did you mean `month == 1`?

sqrt(2)^2==2
#[1] FALSE
> 1/49*49==1
#[1] FALSE

#ao invés de ==,use near()

near(sqrt(2)^2,2)
#[1] TRUE
near(1/49*49,1)
#[1] TRUE


#Para utilizar operadores lógicos (operadores booleanos) (& é "and",| é "or" e ! é "not"):
filter(flights,month==11 | month==12)
#Mas encontrará todos os voos em janeiro

#para corrigir isto use:
nov_dec<-filter(flights,month %in% c(11,12))
