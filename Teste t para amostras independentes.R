
######################### Teste t para Amostras Independentes #########################


# Passo 1: Carregar os pacotes que ser?o usados

if(!require(dplyr)) install.packages("dplyr") # Instala??o do pacote caso n?o esteja instalado
library(dplyr)                                # Carregamento do pacote
if(!require(RVAideMemoire)) install.packages("RVAideMemoire") # Instala??o do pacote caso n?o esteja instalado
library(RVAideMemoire)                                        # Carregamento do pacote
if(!require(car)) install.packages("car") # Instala??o do pacote caso n?o esteja instalado
library(car)                                # Carregamento do pacote

# Passo 2: Carregar o banco de dados

# Importante: selecionar o diret?rio de trabalho (working directory)
# Isso pode ser feito manualmente: Session > Set Working Directory > Choose Directory
# Ou usando a linha de c?digo abaixo:
# setwd("C:/Users/ferna/Desktop")

dados <- read.csv('Banco de Dados 3.csv', sep = ';', dec = ',',
                  stringsAsFactors = T, fileEncoding = "latin1")  # Carregamento do arquivo csvView(dados)                                       # Visualiza??o dos dados em janela separada
glimpse(dados)                                    # Visualiza??o de um resumo dos dados


# Passo 3: Verifica??o da normalidade dos dados
## Shapiro por grupo (pacote RVAideMemoire)

byf.shapiro(Nota_Biol ~ Posicao_Sala, dados)
byf.shapiro(Nota_Fis ~ Posicao_Sala, dados)
byf.shapiro(Nota_Hist ~ Posicao_Sala, dados)

byf.shapiro(Nota_Fis ~ Posicao_Sala, dados)
# Passo 4: Verifica??o da homogeneidade de vari?ncias
## Teste de Levene (pacote car)

leveneTest(Nota_Biol ~ Posicao_Sala, dados, center=mean)
leveneTest(Nota_Fis ~ Posicao_Sala, dados, center=mean)
leveneTest(Nota_Hist ~ Posicao_Sala, dados, center=mean)

# Observa??o:
  # Por default, o teste realizado pelo pacote car tem como base a mediana (median)
    # O teste baseado na mediana ? mais robusto
  # Mudamos para ser baseado na m?dia (compar?vel ao SPSS)


# Passo 5: Realiza??o do teste t para amostras independentes

t.test(Nota_Biol ~ Posicao_Sala, dados, var.equal=TRUE)
t.test(Nota_Fis ~ Posicao_Sala, dados, var.equal=FALSE)
t.test(Nota_Hist ~ Posicao_Sala, dados, var.equal=FALSE)

# Observa??o:
  # O teste bicaudal ? o default; caso deseje unicaudal, necess?rio incluir:
    # alternative = "greater" ou alternative = "less"
  # Exemplo: t.test(Nota_Biol ~ Posicao_Sala, dados, var.equal=TRUE, alternative="greater")
    # Nesse caso, o teste verificar? se ? a m?dia do primeiro grupo ? maior que a m?dia do segundo
      # O R est? considerando "Frente" como primeiro grupo


# Passo 6 (opcional): Visualiza??o da distribui??o dos dados

par(mfrow=c(1,3)) # Estabeleci que quero que os gr?ficos saiam na mesma linha
boxplot(Nota_Biol ~ Posicao_Sala, data = dados, ylab="Notas de Biologia", xlab="Posi??o na Sala")
boxplot(Nota_Fis ~ Posicao_Sala, data = dados, ylab="Notas de F?sica", xlab="Posi??o na Sala")
boxplot(Nota_Hist ~ Posicao_Sala, data = dados, ylab="Notas de Hist?ria", xlab="Posi??o na Sala")

