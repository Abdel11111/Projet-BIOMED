"""si vous voulez visualiser la diffusion dans l'arbre fermez la 1ere fenetre qui apparait lors de l'execution du code
et puis remplissez avec 3 les conditions limites pour chaque sous rectangles des 10 qu'on a soustrait du carre;
si vous vouler visualiser pour le cas d'une autre figure que vous voulez definir commenter la ligne 407 qui permet la constructuion
de l'arbre et ensuite definisez les sous rectangles que vous voulez soustraire par la premiere interface qui s'affiche"""
import cmath
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.animation import FuncAnimation
lambdaa=300#en mm
DX=0.05#le pas de discretisation en mm
r=-lambdaa/(DX+lambdaa)
C=4.7*(10**(-6)) #Ca-Cb en mmol/l
T=5 # durée de respiartion en s
D=19.8 #constante de diffusivite en  mm^2/s
R=D/((DX)**2)
a=1.57 # en s-1 c'est une pulsation
z1 = complex(0, a)
Fn_values=None
figure_points=[]
class SubRectangleEditor:
    #cette classe permet de definir les sous rectangles qu'on veut retrancher du rectangle avec une interface graphique qui va etre afficher une fois le code est executé
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
   
    # Créer une grille discrète selon le pas spécifié
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
   
    # Tri des points pour obtenir l'ordre souhaité (ici, ordonnés par coordonnée y, puis par coordonnée x(on part du bas gauche du rectangle vers la droite et puis on passe vers la ligne des points qui suit en haut))
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
        k = i
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

    # Listes pour stocker les points à gauche, à droite, en bas et en haut de chaque sous-rectangle
    left_points_list, right_points_list, bottom_points_list, top_points_list = [], [], [], []

    # Parcourir chaque sous-rectangle
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
        bottom_points = [(x, y) for x, y in bottom_points if x_min <= x <= x_max and y_min <= y <= y_max]  # Exclure les points en dehors
        bottom_points_list.append(bottom_points)

        # Points en haut de chaque sous-rectangle
        top_points = [(x, y_sub_max + step_size) for x in range(int(x_sub_min), int(x_sub_max) + 1, step_size) if y_sub_max + step_size <= y_max and (x, y_sub_max + step_size) <= (x_max, y_max)]
        top_points.extend([(x_sub_min - step_size, y_sub_max + step_size), (x_sub_max + step_size, y_sub_max + step_size)])  # Ajout des points en haut
        top_points = [(x, y) for x, y in top_points if x_min <= x <= x_max and y_min <= y <= y_max]  # Exclure les points en dehors
        top_points_list.append(top_points)

    return left_points_list, right_points_list, bottom_points_list, top_points_list
def get_boundary_conditions_for_subrectangle(sub_rect_num, has_left, has_right, has_bottom, has_top):
    # Demander à l'utilisateur d'entrer les conditions limites pour le sous-rectangle spécifié
    print(f"Veuillez entrer les conditions limites pour le sous-rectangle {sub_rect_num}:")

    # Conditions pour le côté gauche
    if has_left:
        left_condition = int(input(f"Côté gauche (1 pour Neumann, 2 pour Dirichlet, 3 pour Robin): "))
    else:
        left_condition = 0  # Aucune condition pour ce côté

    # Conditions pour le côté droit
    if has_right:
        right_condition = int(input(f"Côté droit (1 pour Neumann, 2 pour Dirichlet, 3 pour Robin): "))
    else:
        right_condition = 0  # Aucune condition pour ce côté

    # Conditions pour le côté bas
    if has_bottom:
        bottom_condition = int(input(f"Côté bas (1 pour Neumann, 2 pour Dirichlet, 3 pour Robin): "))
    else:
        bottom_condition = 0  # Aucune condition pour ce côté

    # Conditions pour le côté haut
    if has_top:
        top_condition = int(input(f"Côté haut (1 pour Neumann, 2 pour Dirichlet, 3 pour Robin): "))
    else:
        top_condition = 0  # Aucune condition pour ce côté

    # Retourner les conditions limites pour le sous-rectangle
    return left_condition, right_condition, bottom_condition, top_condition
def assign_boundary_conditions(rectangle, sub_rectangles, step_size):
    # Obtenir les listes de points sur les côtés des sous-rectangles
    left_points_list, right_points_list, bottom_points_list, top_points_list = get_points_on_sides(rectangle, sub_rectangles, step_size)

    # Créer un dictionnaire pour stocker les conditions limites pour chaque point
    boundary_conditions = {}

    # Assigner la condition de Neumann (valeur arbitraire de 1) aux points sur le côté gauche
    for point in [(0, x) for x in range(rectangle[1], rectangle[3] + 1, step_size)]:
        boundary_conditions[point] = 1

    # Assigner la condition de Neumann (valeur arbitraire de 1) aux points sur le côté droit
    for point in [(rectangle[2], x) for x in range(rectangle[1], rectangle[3] + 1, step_size)]:
        boundary_conditions[point] = 1

    # Boucle sur les sous-rectangles pour obtenir les conditions limites de l'utilisateur
    for i, sub_rect in enumerate(sub_rectangles, start=1):
        # Déterminer quels côtés sont réels pour ce sous-rectangle
        has_left = sub_rect[0] > rectangle[0]
        has_right = sub_rect[2] < rectangle[2]
        has_bottom = sub_rect[1] > rectangle[1]
        has_top = sub_rect[3] < rectangle[3]

        # Appeler la fonction pour obtenir les conditions limites pour ce sous-rectangle
        left_condition, right_condition, bottom_condition, top_condition = get_boundary_conditions_for_subrectangle(i, has_left, has_right, has_bottom, has_top)

        # Appliquer les conditions limites aux points correspondants sur les côtés des sous-rectangles
        for point in left_points_list[i - 1]:
            boundary_conditions[point] = left_condition
        for point in right_points_list[i - 1]:
            boundary_conditions[point] = right_condition
        for point in bottom_points_list[i - 1]:
            boundary_conditions[point] = bottom_condition
        for point in top_points_list[i - 1]:
            boundary_conditions[point] = top_condition

    # Conditions limites supplémentaires pour y=0 (Rubin) et y=ymax (Dirichlet)
    for point in [(x, 0) for x in range(rectangle[0], rectangle[2] + 1, step_size)]:
        boundary_conditions[point] = 3  # Condition de Robin

    for point in [(x, rectangle[3]) for x in range(rectangle[0], rectangle[2] + 1, step_size)]:
        boundary_conditions[point] = 2  # Condition de Dirichlet

    # Retourner le dictionnaire des conditions limites assignées
    return boundary_conditions
def build_matrix_C(coefficients_dict, neighbors_dict, boundary_conditions,figure_points):
    #la matrice qui permet la resolution de X2
    num_points = len(coefficients_dict)
    matrix_B = np.zeros((num_points, num_points), dtype=complex)

    for k in range(1, num_points + 1):
        # c'est l'equation de diffusion discretisee mais on doit modifier les parties de la matrice concernant les conditions de dirichlet et robin
        matrix_B[k - 1, neighbors_dict[k][0] - 1] =R* coefficients_dict[k]['k1']
        matrix_B[k - 1, neighbors_dict[k][1] - 1] =R* coefficients_dict[k]['k2']
        matrix_B[k - 1, neighbors_dict[k][2] - 1] =R* coefficients_dict[k]['k3']
        matrix_B[k - 1, neighbors_dict[k][3] - 1] =R* coefficients_dict[k]['k4']

        matrix_B[k - 1, k - 1] = R*coefficients_dict[k]['k']-z1 
        # Modifier la ligne k si la condition limite du point k est 2
        condition_k = boundary_conditions.get(figure_points[k - 1], 0)
        if condition_k == 2:
            matrix_B[k - 1, :] = 0
            matrix_B[k - 1, k - 1] = 1 
        # Modifier la ligne k si la condition limite du point k est 3
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
def build_matrix_B(coefficients_dict, neighbors_dict, boundary_conditions,figure_points):
    #la matrice qui permet la resolution du cas stationnaire
    num_points = len(coefficients_dict)
    matrix_B = np.zeros((num_points, num_points))

    for k in range(1, num_points + 1):
        # c'est l'equation de diffusion discretisee mais on doit modifier les parties de la matrice concernant les conditions de dirichlet et robin
        matrix_B[k - 1, neighbors_dict[k][0] - 1] = coefficients_dict[k]['k1']
        matrix_B[k - 1, neighbors_dict[k][1] - 1] = coefficients_dict[k]['k2']
        matrix_B[k - 1, neighbors_dict[k][2] - 1] = coefficients_dict[k]['k3']
        matrix_B[k - 1, neighbors_dict[k][3] - 1] = coefficients_dict[k]['k4']

        matrix_B[k - 1, k - 1] = coefficients_dict[k]['k']
        # Modifier la ligne k si la condition limite du point k est 2
        condition_k = boundary_conditions.get(figure_points[k - 1], 0)
        if condition_k == 2:
            matrix_B[k - 1, :] = 0
            matrix_B[k - 1, k - 1] = 1  
        # Modifier la ligne k si la condition limite du point k est 3
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
    # Tracer un nuage de points en fonction des coordonnées (x, y) avec des couleurs basées sur les conditions aux limites.
    # Chaque point est coloré en fonction de la condition limite associée à ses coordonnées.
    plt.scatter(x_values, y_values, c=[conditions.get((x, y), 0) for x, y in zip(x_values, y_values)])
   
    # Titre du graphique
    plt.title('Discrétisation de l\'espace avec des conditions aux limites')
   
    # Libellé de l'axe X
    plt.xlabel('Axe X')
   
    # Libellé de l'axe Y
    plt.ylabel('Axe Y')
   
    # Afficher la grille sur le graphique
    plt.grid(True)
   
    # Afficher le graphique
    plt.show()
def on_rectangle_select(eclick, erelease):
    #Quand l'utilisateur sélectionne un rectangle ,dans l'interface qui apparait lors de l'execution du code, en cliquant et en relâchant le bouton de la souris. Les coordonnées du coin supérieur gauche et du coin inférieur droit du rectangle sont ensuite stockées dans la variable globale current_rect.

    # Définir la variable globale pour stocker les coordonnées du rectangle sélectionné
    global current_rect
   
    # Enregistrer les coordonnées du coin supérieur gauche (eclick) et du coin inférieur droit (erelease) du rectangle
    current_rect = [eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata]
def plot_heatmap(figure_points, values):
    # Extraire les coordonnées x et y des points de la figure
    x_values, y_values = zip(*figure_points)

    # Obtenir les valeurs uniques de x et y
    unique_x = sorted(set(x_values))
    unique_y = sorted(set(y_values))

    # Créer une grille pour la carte de concentration
    heatmap = np.zeros((len(unique_y), len(unique_x)))

    # Remplir la grille avec les valeurs
    for point, value in zip(figure_points, values):
        x, y = point
        i = unique_y.index(y)
        j = unique_x.index(x)
        heatmap[i, j] = value

    # Définir les extremums de l'échelle des couleurs
    plt.imshow(heatmap, cmap='viridis', extent=[min(unique_x), max(unique_x), min(unique_y), max(unique_y)], origin='lower', vmin=np.min(Fn_values), vmax=np.max(Fn_values))
    plt.colorbar(label='Concentration en mmol/l')
    plt.title('Diffusion de la concentration en O2 dans un acinus')
    plt.xlabel('Longueur 10^(-2)mm')
    plt.ylabel('Longueur 10^(-2)mm')
    plt.grid(True)
    plt.show()
def update(frame):
    plt.clf()  # Effacer la figure actuelle
    plot_heatmap(figure_points, Fn_values[:, frame])
    plt.title(f'Heatmap at Time Step {frame}')
def animate_heatmap():
    # Créer la figure et l'axe
    fig, ax = plt.subplots()

    # Utiliser la fonction d'animation pour mettre à jour la carte de concentration
    animation = FuncAnimation(fig, update, frames=range(300), interval=10)
    plt.show()
def arguments_vecteur_complexes(vecteur):
    # Fonction qui retourne le vecteur dont les coordonnes sont les arguments des coordonnees du vecteur de l'input.
    arguments = [cmath.phase(z) for z in vecteur]
    return arguments
def module_vecteur_complexe(vecteur_complexes):
     # Fonction qui retourne le vecteur dont les coordonnes sont les modules des coordonnees du vecteur de l'input.
    modules = np.abs(vecteur_complexes)
    return modules
def main():
    global Fn_values, figure_points

    # Définition du rectangle et de la taille de pas
    rectangle = (0, 0, 300, 300)  # Unités en 10^(-2)mm
    step_size = 5  # Unités en 10^(-2)mm
    fig, ax = plt.subplots()

    # Utilisation de l'éditeur de sous-rectangles interactif
    sub_rectangle_editor = SubRectangleEditor(ax, rectangle, step_size)

    # Afficher la figure pour permettre à l'utilisateur d'ajouter des sous-rectangles
    plt.show()

    # Récupérer les sous-rectangles ajoutés par l'utilisateur
    sub_rectangles = sub_rectangle_editor.sub_rectangles
    #commenter si vous vouler changez la figure de l'arbre
    sub_rectangles.extend([(10,0,20,150),(40,0,50,150),(70,0,80,150),(100,0,110,150),(130,0,140,150),(160,0,170,150),(190,0,200,150),(220,0,230,150),(250,0,260,150),(280,0,290,150)])
    # Afficher les sous-rectangles ajoutés
    print("Sous-rectangles ajoutés:")
    for sub_rect in sub_rectangles:
        print(sub_rect)

    # Discrétiser l'espace en points, définir les conditions aux limites et construire les matrices
    figure_points, neighbors_dict, coefficients_dict = discretize_space(rectangle, sub_rectangles, step_size)
    boundary_conditions = assign_boundary_conditions(rectangle, sub_rectangles, step_size)
    matrix_C = build_matrix_C(coefficients_dict, neighbors_dict, boundary_conditions, figure_points)
    matrix_B = build_matrix_B(coefficients_dict, neighbors_dict, boundary_conditions, figure_points)

    # Calculer les matrices inverses
    inverse_matrix_C = np.linalg.inv(matrix_C)
    inverse_matrix_B = np.linalg.inv(matrix_B)

    # Créer des vecteurs et matrices nécessaires pour la résolution du système
    vector_1=np.zeros(len(figure_points))
    vector_with_ones = np.zeros(len(figure_points))
    vector_eq = np.zeros(len(figure_points))
    #cette boucle permet de calculer la concentration C1 qui est la moitie de l'ecart entre les extremums de la concentration dans la figure (donc depend fatalement de la figure)
    for i, point in enumerate(figure_points, start=1):
        if boundary_conditions.get(point, 0) == 2:
            vector_1[i - 1] = C
    result_vector1 = np.dot(inverse_matrix_B, vector_1)
    C1=  (result_vector1[len(figure_points)-1]-result_vector1[1])/2 
 

    # Identifier les points avec condition limite 2
    for i, point in enumerate(figure_points, start=1):#le i permet d'ennumerer les points
        if boundary_conditions.get(point, 0) == 2:
            vector_with_ones[i - 1] = C1
            vector_eq[i - 1] = C - C1

    # Résoudre le système linéaire avec les matrices inverses
    result_vector = np.dot(inverse_matrix_C, vector_with_ones) # les coordonnees k de ce vecteur sont les valeurs de la grandeur X2 dans les points k (il est complexe)
    result_vector2 = np.dot(inverse_matrix_B, vector_eq) # les coordonnees k de ce vecteur sont les valeurs de la grandeur X dans les points k (c'est comme le vecteur de resolution dans le cas stationnaire ou la condition d'entree c'est X=C)

    # Calculer les arguments et les modules des vecteurs complexes
    arguments = arguments_vecteur_complexes(result_vector)
    module = module_vecteur_complexe(result_vector)

    # Initialiser la matrice Fn_values pour stocker les valeurs de la grandeur X dans le temps ;Chaque colonne k de la matrice representera les valeur de X pour l'instant discretisé k.
    Fn_values = np.zeros((len(figure_points), 300))

    # Remplissage de Fn_values
    DT = T / 300 #pas de descritation du temps / T est la duree de la diffusion
    for n in range(300):#boucle sur les instants on a pris 300 instants dans ce cas
        for k in range(len(figure_points)):
            Fn_values[k, n] = module[k] * np.cos(a * n * DT + arguments[k]) + result_vector2[k] #(combinaison des solutions du cas stationnaire et du cas frequentiel )

    # Récupérer les coordonnées x, y des points de la figure
    x_values, y_values = zip(*figure_points)

    # Afficher le graphique des points avec les conditions aux limites
    plot_points_with_conditions(x_values, y_values, boundary_conditions)

    # Afficher l'animation de la carte thermique
    animate_heatmap()
if __name__ == "__main__":
    main()