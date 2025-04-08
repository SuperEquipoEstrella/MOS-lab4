import numpy as np
from tabulate import tabulate

def simplex_con_iteraciones(A, b, c, imprimir=True, modo='max'):
    """
    Implementa el Método Simplex y retorna los valores finales 
    de todas las variables (originales y de holgura o artificiales).
    """
    m, n = A.shape

    if modo == 'min':
        c = -c

    # Matriz extendida con variables de holgura o artificiales
    I = np.eye(m)
    A_ext = np.hstack([A, I])
    c_ext = np.concatenate([c, np.zeros(m)])

    matriz_simplex = np.zeros((m + 1, n + m + 1))

    for i in range(m):
        matriz_simplex[i, :-1] = A_ext[i]
        matriz_simplex[i, -1] = b[i]

    matriz_simplex[-1, :-1] = -c_ext

    iteracion = 0
    while True:
        if imprimir:
            print(f"\nIteración {iteracion}:")
            print(tabulate(matriz_simplex, floatfmt=".3f", tablefmt="grid"))
            print("-" * 50)

        fila_objetivo = matriz_simplex[-1, :-1]
        col_in = np.argmin(fila_objetivo)
        valor_min = fila_objetivo[col_in]

        if valor_min >= 0:
            if imprimir:
                print("Óptimo alcanzado.")
            break

        columna_pivote = matriz_simplex[:-1, col_in]
        b_column = matriz_simplex[:-1, -1]
        razones = []

        for i in range(m):
            if columna_pivote[i] > 1e-15:
                razones.append(b_column[i] / columna_pivote[i])
            else:
                razones.append(np.inf)

        fila_out = np.argmin(razones)
        if razones[fila_out] == np.inf:
            raise ValueError("El problema es no acotado.")

        if imprimir:
            print(f"Columna que entra: x{col_in+1}")
            print(f"Fila que sale: {fila_out+1}")
            print(f"Valor pivote: {matriz_simplex[fila_out, col_in]:.4f}")

        pivote = matriz_simplex[fila_out, col_in]
        matriz_simplex[fila_out, :] /= pivote

        for i in range(m + 1):
            if i != fila_out:
                factor = matriz_simplex[i, col_in]
                matriz_simplex[i, :] -= factor * matriz_simplex[fila_out, :]

        iteracion += 1

    z_opt = matriz_simplex[-1, -1]
    if modo == 'min':
        z_opt = -z_opt

    return matriz_simplex, z_opt


def extraer_resultados_simplex(matriz_simplex, A):
    m, n = A.shape
    x_opt = np.zeros(n + m)
    for col_index in range(n + m):
        col = matriz_simplex[:m, col_index]
        if np.count_nonzero(col) == 1 and np.isclose(np.max(col), 1.0):
            row_index = np.argmax(col)
            x_opt[col_index] = matriz_simplex[row_index, -1]

    resultados = {}
    for i in range(n):
        resultados[f'x{i+1}'] = x_opt[i]
    for i in range(m):
        resultados[f's{i+1}'] = x_opt[n + i]
    return resultados


def fase_1():
    """
    Ejecuta la Fase I del Método de Dos Fases para el problema dado.
    """
    # Matriz A con variables: [x1, x2, x3, s1, e1, a0, a1]
    A_aux = np.array([
        [ 2,  1, -1,  0,  0, 1, 0],  # R1: 2x1 + x2 - x3 + a0 = 10
        [ 1, -3,  2,  0, -1, 0, 1], # R2: x1 - 3x2 + 2x3 - e1 + a1 = 5
        [ 1,  1,  1,  1,  0, 0, 0]  # R3: x1 + x2 + x3 + s1 = 15
    ], dtype=float)

    b = np.array([10, 5, 15], dtype=float)

    # Función objetivo auxiliar: w = a0 + a1 ⇒ max(-a0 - a1)
    c_aux = np.array([0, 0, 0, 0, 0, -1, -1], dtype=float)

    matriz_final, w_opt = simplex_con_iteraciones(A_aux, b, c_aux, imprimir=True, modo='max')

    print("\nValor óptimo del problema auxiliar (w*):", w_opt)
    if abs(w_opt) < 1e-8:
        print("✅ Se encontró una SBF para el problema original.")
        resultados = extraer_resultados_simplex(matriz_final, A_aux)
        print("Variables en la solución básica:")
        for var, val in resultados.items():
            print(f"  {var} = {val:.4f}")
    else:
        print("❌ El problema original es infactible.")


# Ejecutar la Fase I
fase_1()
