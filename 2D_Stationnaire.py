"""Pour observer la diffusion au sein de l'arbre, veuillez decommenter la ligne 305 du code. Ensuite, fermez la première fenêtre
qui apparaît lors de l'exécution du programme. Subséquemment, attribuez la valeur 3 aux conditions limites de chaque sous-rectangle
parmi les dix qui ont été soustraits du carré initial// Si vous souhaitez visualiser la diffusion pour une autre figure
que vous souhaitez définir, veuillez commenter la ligne 305 responsable de la construction de l'arbre.
Ensuite, définissez les sous-rectangles que vous souhaitez soustraire via la première interface qui s'affiche."""
#-fermer la deuxième fenêtre qui apparaitra qui représente l’affichage des points avec leurs conditions aux limites
#-Le résultat s’affichera a la 3ème fenêtre
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
lambdaa=300#en mm
DX=0.05#le pas de discretisation en mm
r=-lambdaa/(DX+lambdaa)
C=4.7*(10**(-6))#Ca-Cb en mmol/l
L=3#longueur de l'arret de l'acinus modelisé en cube
D=19.8 #constante de diffusivite en  mm^2/s

class SubRectangleEditor:
    def __init__(self, ax, rectangle, step_size):
        self.ax = ax
        self.rectangle = rectangle
        self.step_size = step_size
        self.sub_rectangles = []
       
        self.ax.set_xlim(rectangle[0]-5, rectangle[2]+5)
        self.ax.set_ylim(rectangle[1]-5, rectangle[3]+5)
        self.ax.set_title('Cliquez successivement sur deux points pour définir un sous-rectangle')

        self.ax.figure.canvas.mpl_connect('button_press_event', self.onclick)
   
    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if not hasattr(self, 'point1'):
            self.point1 = (int(x / self.step_size) * self.step_size, int(y / self.step_size) * self.step_size)
            self.ax.scatter(*self.point1, color='red')
        else:
            point2 = (int(x / self.step_size) * self.step_size, int(y / self.step_size) * self.step_size)
            self.ax.scatter(*point2, color='red')
            self.add_sub_rectangle(self.point1, point2)
            delattr(self, 'point1')
            self.ax.figure.canvas.draw()

    def add_sub_rectangle(self, point1, point2):
        x_min, y_min = min(point1[0], point2[0]), min(point1[1], point2[1])
        x_max, y_max = max(point1[0], point2[0]), max(point1[1], point2[1])

        sub_rectangle = (x_min, y_min, x_max, y_max)
        self.sub_rectangles.append(sub_rectangle)
        self.draw_rectangle(sub_rectangle)
   
   
    def draw_rectangle(self, sub_rectangle):
        rect = Rectangle(
            (sub_rectangle[0], sub_rectangle[1]),
            sub_rectangle[2] - sub_rectangle[0],
            sub_rectangle[3] - sub_rectangle[1],
            fill=None,
            edgecolor='red'
        )
        self.ax.add_patch(rect)
def discretize_space(rectangle, sub_rectangles, step_size):
    # Définir les limites du rectangle initial
    x_min, y_min, x_max, y_max = rectangle
   
    # Créer une grille discrète
    x_values = list(range(int(x_min), int(x_max) + 1, step_size))
    y_values = list(range(int(y_min), int(y_max) + 1, step_size))
   
    # Créer une liste pour stocker les points de la figure résultante
    figure_points = []
   
    # Ajouter les points du rectangle initial à la liste
    for x in x_values:
        for y in y_values:
            figure_points.append((x, y))
   
    # Soustraire les points des sous-rectangles
    for sub_rect in sub_rectangles:
        x_sub_min, y_sub_min, x_sub_max, y_sub_max = sub_rect
        for x in range(int(x_sub_min), int(x_sub_max) + 1, step_size):
            for y in range(int(y_sub_min), int(y_sub_max) + 1, step_size):
                if (x, y) in figure_points:
                    figure_points.remove((x, y))
   
    # Tri des points pour obtenir l'ordre souhaité
    figure_points.sort(key=lambda p: (p[1], p[0]))
   
    # Créer un dictionnaire pour stocker les voisins de chaque point
    neighbors_dict = {}
   
    # Trouver les voisins de chaque point
    for i, point in enumerate(figure_points, start=1):
        neighbors = []
        x, y = point
       
        # Voisin à gauche
        left_neighbor = (x - step_size, y)
        if left_neighbor in figure_points:
            neighbors.append(figure_points.index(left_neighbor) + 1)
        else:
            neighbors.append(0)
       
        # Voisin à droite
        right_neighbor = (x + step_size, y)
        if right_neighbor in figure_points:
            neighbors.append(figure_points.index(right_neighbor) + 1)
        else:
            neighbors.append(0)
       
        # Voisin en dessous
        below_neighbor = (x, y - step_size)
        if below_neighbor in figure_points:
            neighbors.append(figure_points.index(below_neighbor) + 1)
        else:
            neighbors.append(0)
       
        # Voisin au-dessus
        above_neighbor = (x, y + step_size)
        if above_neighbor in figure_points:
            neighbors.append(figure_points.index(above_neighbor) + 1)
        else:
            neighbors.append(0)
       
        neighbors_dict[i] = neighbors
   
    # Calculer les coefficients pour chaque point
    coefficients_dict = {}
    for i, point in enumerate(figure_points, start=1):
        x, y = point
        k1, k2, k3, k4 = neighbors_dict[i]
        k_neighbors = [n for n in neighbors_dict[i] if n != 0]
        coefficients_dict[i] = {
            'k1': 1 if k1 != 0 else 0,
            'k2': 1 if k2 != 0 else 0,
            'k3': 1 if k3 != 0 else 0,
            'k4': 1 if k4 != 0 else 0,
            'k': -len(k_neighbors)
        }
    return figure_points, neighbors_dict, coefficients_dict
def get_points_on_sides(rectangle, sub_rectangles, step_size):
    x_min, y_min, x_max, y_max = rectangle
   
    # Points à gauche, à droite, en bas et en haut de chaque sous-rectangle
    left_points_list, right_points_list, bottom_points_list, top_points_list = [], [], [], []
   
    for sub_rect in sub_rectangles:
        x_sub_min, y_sub_min, x_sub_max, y_sub_max = sub_rect
       
        # Points à gauche de chaque sous-rectangle
        left_points = [(x_sub_min - step_size, y) for y in range(int(y_sub_min), int(y_sub_max) + 1, step_size) if x_sub_min - step_size >= x_min and (x_sub_min - step_size, y) <= (x_max, y_max)]
        left_points_list.append(left_points)
       
        # Points à droite de chaque sous-rectangle
        right_points = [(x_sub_max + step_size, y) for y in range(int(y_sub_min), int(y_sub_max) + 1, step_size) if x_sub_max + step_size <= x_max and (x_sub_max + step_size, y) <= (x_max, y_max)]
        right_points_list.append(right_points)
       
        # Points en bas de chaque sous-rectangle
        bottom_points = [(x, y_sub_min - step_size) for x in range(int(x_sub_min), int(x_sub_max) + 1, step_size) if y_sub_min - step_size >= y_min and (x, y_sub_min - step_size) <= (x_max, y_max)]
        bottom_points.extend([(x_sub_min - step_size, y_sub_min - step_size), (x_sub_max + step_size, y_sub_min - step_size)])  # Ajout des points en bas
       
        bottom_points_list.append(bottom_points)
       
        # Points en haut de chaque sous-rectangle
        top_points = [(x, y_sub_max + step_size) for x in range(int(x_sub_min), int(x_sub_max) + 1, step_size) if y_sub_max + step_size <= y_max and (x, y_sub_max + step_size) <= (x_max, y_max)]
        top_points.extend([(x_sub_min - step_size, y_sub_max + step_size), (x_sub_max + step_size, y_sub_max + step_size)])  # Ajout des points en haut

        top_points_list.append(top_points)
   
    return left_points_list, right_points_list, bottom_points_list, top_points_list
def get_boundary_conditions_for_subrectangle(sub_rect_num, has_left, has_right, has_bottom, has_top):
    print(f"Veuillez entrer les conditions limites pour le sous-rectangle {sub_rect_num}:")
   
    if has_left:
        left_condition = int(input(f"Côté gauche (1 pour Neumann, 2 pour Dirichlet, 3 pour Robin): "))
    else:
        left_condition = 0  # Aucune condition pour ce côté
   
    if has_right:
        right_condition = int(input(f"Côté droit (1 pour Neumann, 2 pour Dirichlet, 3 pour Robin): "))
    else:
        right_condition = 0  # Aucune condition pour ce côté
   
    if has_bottom:
        bottom_condition = int(input(f"Côté bas (1 pour Neumann, 2 pour Dirichlet, 3 pour Robin): "))
    else:
        bottom_condition = 0  # Aucune condition pour ce côté
   
    if has_top:
        top_condition = int(input(f"Côté haut (1 pour Neumann, 2 pour Dirichlet, 3 pour Robin): "))
    else:
        top_condition = 0  # Aucune condition pour ce côté
   
    return left_condition, right_condition, bottom_condition, top_condition
def assign_boundary_conditions(rectangle, sub_rectangles, step_size):
    left_points_list, right_points_list, bottom_points_list, top_points_list = get_points_on_sides(rectangle, sub_rectangles, step_size)
   
    # Create a dictionary to store boundary conditions for each point
    boundary_conditions = {}
    for point in [(0, x) for x in range(rectangle[0], rectangle[3] + 1, step_size)]:
        boundary_conditions[point] = 1
    for point in [(rectangle[2], x) for x in range(rectangle[0], rectangle[3] + 1, step_size)]:
        boundary_conditions[point] = 1
   
    for i, sub_rect in enumerate(sub_rectangles, start=1):
        # Déterminer quels côtés sont réels pour ce sous-rectangle
        has_left = sub_rect[0] > rectangle[0]
        has_right = sub_rect[2] < rectangle[2]
        has_bottom = sub_rect[1] > rectangle[1]
        has_top = sub_rect[3] < rectangle[3]
       
        # Appeler la fonction modifiée avec les informations sur les côtés réels
        left_condition, right_condition, bottom_condition, top_condition = get_boundary_conditions_for_subrectangle(i, has_left, has_right, has_bottom, has_top)
       
        # Appliquer les conditions aux limites aux points correspondants
        for point in left_points_list[i - 1]:
            boundary_conditions[point] = left_condition
        for point in right_points_list[i - 1]:
            boundary_conditions[point] = right_condition
        for point in bottom_points_list[i - 1]:
            boundary_conditions[point] = bottom_condition
        for point in top_points_list[i - 1]:
            boundary_conditions[point] = top_condition
    # Additional conditions for y=0 (Rubin) and y=ymax (Dirichlet)
    for point in [(x, 0) for x in range(rectangle[0], rectangle[2] + 1, step_size)]:
        boundary_conditions[point] = 3 # Rubin condition
   
    for point in [(x, rectangle[3]) for x in range(rectangle[0], rectangle[2] + 1, step_size)]:
        boundary_conditions[point] = 2  # Dirichlet condition
    return boundary_conditions
def build_matrix_B(coefficients_dict, neighbors_dict, boundary_conditions,figure_points):
    num_points = len(coefficients_dict)
    matrix_B = np.zeros((num_points, num_points))

    for k in range(1, num_points + 1):
        # Soustrayez 1 car les indices commencent à 1
        matrix_B[k - 1, neighbors_dict[k][0] - 1] = coefficients_dict[k]['k1']
        matrix_B[k - 1, neighbors_dict[k][1] - 1] = coefficients_dict[k]['k2']
        matrix_B[k - 1, neighbors_dict[k][2] - 1] = coefficients_dict[k]['k3']
        matrix_B[k - 1, neighbors_dict[k][3] - 1] = coefficients_dict[k]['k4']

        matrix_B[k - 1, k - 1] = coefficients_dict[k]['k']
        # Modifier la ligne k si la condition limite du point k est 2 ou 3
        condition_k = boundary_conditions.get(figure_points[k - 1], 0)
        if condition_k == 2:
            matrix_B[k - 1, :] = 0
            matrix_B[k - 1, k - 1] = 1  
        if condition_k == 3 and coefficients_dict[k]['k']!=-4 and coefficients_dict[k]['k']!=-2 :
                matrix_B[k - 1, neighbors_dict[k][0] - 1] = r * (coefficients_dict[k]['k1'] * coefficients_dict[k]['k3'] * coefficients_dict[k]['k4'])
                matrix_B[k - 1, neighbors_dict[k][1] - 1] = r * (coefficients_dict[k]['k2'] * coefficients_dict[k]['k3'] * coefficients_dict[k]['k4'])
                matrix_B[k - 1, neighbors_dict[k][2] - 1] = r * (coefficients_dict[k]['k3'] * coefficients_dict[k]['k2'] * coefficients_dict[k]['k1'])
                matrix_B[k - 1, neighbors_dict[k][3] - 1] = r * (coefficients_dict[k]['k4'] * coefficients_dict[k]['k2'] * coefficients_dict[k]['k1'])
                matrix_B[k - 1, k - 1] = 1 
        if condition_k == 3 and coefficients_dict[k]['k']==-2 :
                matrix_B[k - 1, neighbors_dict[k][0] - 1] = 0
                matrix_B[k - 1, neighbors_dict[k][1] - 1] = 0
                matrix_B[k - 1, neighbors_dict[k][2] - 1] = r *coefficients_dict[k]['k3']
                matrix_B[k - 1, neighbors_dict[k][3] - 1] = r *coefficients_dict[k]['k4']
                matrix_B[k - 1, k - 1] = 1 
       
    return matrix_B
def plot_points_with_conditions(x_values, y_values, conditions):
    plt.scatter(x_values, y_values, c=[conditions.get((x, y), 0) for x, y in zip(x_values, y_values)])
    plt.title('Discretization of Space with Boundary Conditions')
    plt.xlabel('en 10-2mm')
    plt.ylabel('en 10-2mm')
    plt.grid(True)
    plt.show()
def on_rectangle_select(eclick, erelease):
    global current_rect
    current_rect = [eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata]
def plot_concentrationmap(figure_points, values):
    x_values, y_values = zip(*figure_points)
    unique_x = sorted(set(x_values))
    unique_y = sorted(set(y_values))
   
    # Create a grid for the heatmap
    heatmap = np.zeros((len(unique_y), len(unique_x)))
   
    # Fill the grid with the values
    for point, value in zip(figure_points, values):
        x, y = point
        i = unique_y.index(y)
        j = unique_x.index(x)
        heatmap[i, j] = value
   
    # Set vmin and vmax to fix the color scale
    plt.imshow(heatmap, cmap='viridis', extent=[min(unique_x), max(unique_x), min(unique_y), max(unique_y)], origin='lower',vmin=np.min(values), vmax=np.max(values))
    plt.colorbar(label='Concentration en mmol/l')
    plt.title('Concentration en O2 ')
    plt.xlabel('Longueur 10-2mm')
    plt.ylabel('Longueur 10-2mm')
    plt.grid(True)
    plt.show()
def main():
    rectangle = (0, 0, 300, 300) #les unitees sont en 10^(-2)
    step_size = 5
    fig, ax = plt.subplots()
    sub_rectangle_editor = SubRectangleEditor(ax, rectangle, step_size)
    plt.show()

    sub_rectangles = sub_rectangle_editor.sub_rectangles
    #decommenter pour avoir la figure de l'arbre qu'on a étudie dans le rapport
    #sub_rectangles.extend([(10,0,20,150),(40,0,50,150),(70,0,80,150),(100,0,110,150),(130,0,140,150),(160,0,170,150),(190,0,200,150),(220,0,230,150),(250,0,260,150),(280,0,290,150)])
    print("Sous-rectangles ajoutés:")
    for sub_rect in sub_rectangles:
        print(sub_rect)
    figure_points, neighbors_dict, coefficients_dict = discretize_space(rectangle, sub_rectangles, step_size)
    boundary_conditions = assign_boundary_conditions(rectangle, sub_rectangles, step_size)
    matrix_B = build_matrix_B(coefficients_dict, neighbors_dict,boundary_conditions,figure_points)
    inverse_matrix_B = np.linalg.inv(matrix_B)
    vector_with_ones = np.zeros(len(figure_points))

    # Identifier les points avec condition limite 2
    for i, point in enumerate(figure_points, start=1):
        if boundary_conditions.get(point, 0) == 2:
            vector_with_ones[i - 1] = C
    result_vector = np.dot(inverse_matrix_B, vector_with_ones)

    x_values, y_values = zip(*figure_points)
    plot_points_with_conditions(x_values, y_values, boundary_conditions)  
    plot_concentrationmap(figure_points, result_vector)
    #calcul du debit pour la figure considerée par l'utilisateur
    l=0
    num_points=len(figure_points)
    for k in range(1, num_points + 1):

        condition_k = boundary_conditions.get(figure_points[k - 1], 0)

        if condition_k==3:

            l=l+(result_vector[k-1])*D*DX*22.4*L/lambdaa
    print('valeur du debit correpondant a la figure choisie en ml/s')
    print(l)

if __name__ == "__main__":
    main()