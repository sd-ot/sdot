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
