# Se o pacote ainda não estiver instalado:
# install.packages("gee")
library(gee)
# Modelo com uma variável preditiva
modelo <- gee(resposta ~ preditiva, data = seus_dados, family = gaussian, corstr = "exchangeable", id = "id_cluster")

# Modelo com múltiplas variáveis preditivas
modelo <- gee(resposta ~ preditiva1 + preditiva2, data = seus_dados, family = binomial, corstr = "unstructured", id = "id_cluster")
library(gee)

# Criar dados de exemplo
set.seed(123)
n <- 100  # Número de pacientes
t <- rep(1:5, each = n/5)  # Tempo de medição
tratamento <- rep(c("A", "B"), each = n/2)  # Tratamento
id_cluster <- rep(1:n/5, each = 5) # Identificadores dos pacientes
pres_arterial <- rnorm(n, mean = 120, sd = 10) + 
rnorm(n, mean = 0, sd = 2)*t + rnorm(n, mean = 0, sd = 3)*(tratamento == "B") # Simulação da pressão arterial
dados <- data.frame(id_cluster, t, tratamento, pres_arterial)

# Ajustar o modelo GEE
modelo_gee <- gee(pres_arterial ~ t + tratamento, data = dados, family = gaussian, corstr = exchangeable, id = id_cluster)

# Ver os resultados
summary(modelo_gee)
