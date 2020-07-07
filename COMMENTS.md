import pysdot

positions = np.random.rand(200, 2)
# ou une fonction qui prend une fonction pour donner les poids, les masses
#  positions = lambda f:
#    f( np.random.rand(200, 2) )
#    f( np.random.rand(300, 2) )
#    ...

# obligation de donner la densité, forcer de donner des poids

lag = pysdot.solve_optimal_transport(positions, masses, pysdot.densities.box( ... ), lag, positions_have_changed = False) # ou adjust_kantorovith_potentials

for i in ...
    lag.positions[0,i] += 1 # ou immutable, avec set_positions qui met à jour date

lag = pysdot.solve_optimal_transport(lag, pysdot.densities.box( ... )) # ou adjust_kantorovith_potentials


lag = pysdot.optimal_transport(positions, pysdot.densities.box( ... ) ) # ou adjust_kantorovith_potentials
lag.potentials

lag.masses(density)
lag.barycenters(density)
lag.variances(density) # \int (x-xi)^2

# LaguerreWithDensity

lag.gradient_of_masses_wrt_potentials()
lag.gradient_of_masses_wrt_positions()

with_density( local_density = "...",  )

# Voir si ce qu'on fait est compatible avec PyTorch, TF, ...

# ou
# ot = pysdot.transport_plan()
# pysdot.adjust_dirac_weights(ot, positions)
# 
# ou
# ot = pysdot.transport_plan()
# ot.positions = ...
# ot.adjust_dirac_weights()

# pour éviter les recalculs
# ot.adjust_dirac_weights(positions_have_changed = False)
# pysdot.adjust_dirac_weights(ot, positions_have_changed = False)

ot.display()
print(ot.dirac_weights)


# pour le out-of-core et MPI
for chunk in ot.all_the_local_chunks:
    ot.set_active_chunks( [ chunk ] )
    print(ot.dirac_weights)
    # print( chunk.nb_diracs )
    # ...

    # pour les données voisines
    print(ot.get_dirac_weights( other_chunk ))


# ou
for _ in ot.for_each_chunk:
    ot.set

# données additionnelles. Si les chunks sont assez gros, on peut tolérer des données dynamiques
# On pourrait les nommer ou les indexer. Prop: en C++ en interne, c'est indexé, mais il y a une map pour trouver fonction du nom


# Rq: statégie de partitionnement. Proposition actuelle: on fait un histogramme / zindex puis un scan. Pb d'heuristique potentiel.
# Il 
# Si tout loge dans un chunk, on ne va pas chercher plus loin.

# Pour le 
API qui ressemble à Voro++. Examples pour comparer. 
  => on pertube les positions, influence sur le volume.
  Courant d'intégration.






* faire les grilles, avec le out of core.
    L'approche proposée, c'est de faire des grosses sous-structures qui logent en mémoire pour la construction initiale: ça permet de répartir les tri radix sur les différentes machines.
      On a intérêt 
    Cependant, on a besoin d'une granularité 


    Si tout loge en mémoire, on fait juste un tri

    Rq: on a besoin de copier les coordonnées, les poids et les ids pour y faire un tri.

    Si on a tous les 

* Version de base:
  * on fait le min et le max
  * on fait un histogramme avec taille = 2^{...} (nb diracs vs zcoords)
  * on fait un scan de l'histogramme
  * on sépare en deux récursivement le scan pour faire un octree jusqu'aux sous-structures où nb diracs <= 30 (on ne les rempli pas à ce stade, c'est juste pour le dimensionnement)
    On fait scan_ou_hist[ zcoords ] => num sst
    Si on se retrouve avec des sous-structures trop grosses, on refait des parcourts avec des histogrammes locaux...

Les histogrammes, c'est cool quand on a une répartition uniforme. Sinon, il vaut mieux faire un tri. Peut-on faire un tri radix distribué ?
  A priori, oui: on fait le compte pour chaque digit dans des listes séparées. On fait le dispatch dans une seul machine, etc...
  Le dispatch dit vers où envoyer les données ? On pourrait 

Le tri semble difficile. L'idée pourrait être de faire un histogramme pour la répartition en grosses sous-structures. Autrement dit: on se sert de l'histogramme autant que possible. Si on peut commencer à faire des sous-structures avec, c'est cool (on n'aura pas forcément à trier). La taille de l'histogramme est bornée en RAM, ce qui fait que toutes les machines le connaissent.
  Si l'histogramme permet de répartir complétement dans des machines (avec des numéros internes de sous-structures), c'est cool.
  Si l'histogramme contient des nb_diracs trop importants pour que ça loge en RAM, on peut refaire un passage pour se focaliser sur ces parties là: on peut faire un histogramme récursif... mais bon, à la fin, la quantité de RAM nécessaire pour stocker les histogrammes cesse d'être bornée. Rq: à ce stade, c'est juste pour que les diracs logent en RAM... ça devrait passer.

  On peut ensuite refaire un passage pour envoyer les diracs aux machines. Lorsque les machines reçoivent des diracs, elles les collent dans les sous-structures correspondantes.

  L'étape d'après consiste à diviser les sous-structures trop grosses. On peut les faire avec les données copiées

* min_max
* histo au premier degré
* reservation des boites intermédiaires et finales. Certaines boites finales pourraient être trop grosses.
  La solution telle qu'écrit jusqu'à présent, c'est de refaire un passage:
    * ... mais on verra plus tard
* pour la zone en cours (z)
  * remplissage des boites 
  * tri optionnel dans les boites, min et max pts réels
  * représentation synthétique des poids
    Pb: ça serait bien d'avoir le choix du degré du polynôme

Dans for_each_laguerre_cell, on veut récupérer la cellule de Laguerre, la pos, le poids et l'id du Dirac, mais aussi les données du gradient conjugué s'il y en a, ...
  L'idée, c'est de mettre les données pos poids, id et autres au même endroit pour être capables de gérer le MPI et le out of core
  Prop 1: 



nb_cut_case = [ nb_pour_cas_0, nb_pour_cas_1, ...pour chaque cas de coupe ] qui démarre à 0
  À chaque test, on récupère nc = [ num_cas_item_0, num_cas_item_1, ... ]
  S'il n'y a pas de collision, 
    VI ncc = VI::gather( nb_cut_case.data, nc ); // [ nb_cut_case_pour_item_0, ... ]
    VI::scatter( sc[ CutCase() ][ i ].data, ncc ); // on stocke les index
    ncc += 1; // on ajoute 1
    VI::scatter( nb_cut_case.data, nc, ncc ); // on met à jour la taille de chaque list

Prop: la liste d'indices pour chaque cas est une liste contigue, avec une séparation en puissance de 2 entre chaque sous-liste. On stocke un nombre d'indices pour chaque cas
    VI nbi = VI::gather( nb_indices_for_each_case.data, nc );
    VI::scatter( liste_d'indices.data, ( nc << taille_nb_elems ) + nbi ); // on stocke les index
    VI::scatter( nb_indices_for_each_case.data, nc, nbi + 1 );

Pour gérer les collisions. Exemple avec nc = [ 4, 2, 4, 1 ] (4 apparaît 2 foix)
    VI nbi = VI::gather( nb_indices_for_each_case.data, nc ); // -> [ nb_inds_cas_4, nb_inds_cas_2, nb_inds_cas_4, nb_inds_cas_1 ]
    nbi += nb_cas_similaires_au_cas_courant_dans_le_batch.
    VI::scatter( liste_d'indices.data, ( nc << taille_nb_elems ) + nbi ); // on stocke les index
    // -> prendre le max de chaque cas dans nbi + 1
    VI::scatter( nb_indices_for_each_case.data, nc, nbi );





