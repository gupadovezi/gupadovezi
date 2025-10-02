/* ControleTemperaturas.c
   Exemplo para atividade: usa vetor dinâmico, laços, decisões e saída formatada.
   Compile: gcc ControleTemperaturas.c -o ControleTemperaturas
   Execute: ./ControleTemperaturas
*/

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n;
    float *temp = NULL;
    float soma = 0.0f;
    float media;
    float max_val, min_val;
    int dia_max = 0, dia_min = 0;
    int i, acima_media = 0;

    printf("=== Controle de Temperaturas (Vetor[n]) ===\n");
    printf("Quantos dias deseja registrar? ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Entrada invalida. Informe um numero inteiro positivo.\n");
        return 1;
    }

    /* Aloca vetor dinamico de n floats */
    temp = (float*) malloc(sizeof(float) * n);
    if (temp == NULL) {
        fprintf(stderr, "Erro de memoria.\n");
        return 1;
    }

    /* Leitura das temperaturas */
    for (i = 0; i < n; ++i) {
        printf("Dia %d - temperatura maxima (°C): ", i + 1);
        if (scanf("%f", &temp[i]) != 1) {
            fprintf(stderr, "Entrada invalida. Abortando.\n");
            free(temp);
            return 1;
        }
        soma += temp[i];
    }

    /* Calcula media, max e min */
    media = soma / n;
    max_val = min_val = temp[0];
    dia_max = dia_min = 1;
    for (i = 1; i < n; ++i) {
        if (temp[i] > max_val) {
            max_val = temp[i];
            dia_max = i + 1;
        }
        if (temp[i] < min_val) {
            min_val = temp[i];
            dia_min = i + 1;
        }
    }

    /* Conta quantos dias ficaram acima da media */
    for (i = 0; i < n; ++i) {
        if (temp[i] > media) acima_media++;
    }

    /* Saida bem formatada */
    printf("\n--- Resumo das Temperaturas ---\n");
    printf("Media das temperaturas: %.2f °C\n", media);
    printf("Maior temperatura: %.2f °C (Dia %d)\n", max_val, dia_max);
    printf("Menor temperatura: %.2f °C (Dia %d)\n", min_val, dia_min);
    printf("Dias acima da media: %d de %d\n", acima_media, n);

    printf("\nLista completa de temperaturas:\n");
    for (i = 0; i < n; ++i) {
        printf("Dia %2d: %6.2f °C\n", i + 1, temp[i]);
    }

    /* Sugestao simples de "grafico" ASCII: barra proporcional */
    printf("\nGrafico ASCII (barra horizontal proporcional):\n");
    for (i = 0; i < n; ++i) {
        int barras = (int)(temp[i] - min_val); /* simples proporcionalidade */
        if (barras < 0) barras = 0;
        printf("D%2d [%5.2f°C]: ", i+1, temp[i]);
        for (int b = 0; b < barras; ++b) putchar('*');
        putchar('\n');
    }

    free(temp);
    return 0;
}
