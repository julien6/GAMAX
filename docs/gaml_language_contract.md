# Contrat GAML effectivement pris en charge

Ce contrat décrit le compilateur de développement du 14 octobre 2026. Il ne promet pas la compatibilité avec tout GAMA. Le manifeste émis conserve les hashes de l'implémentation, les capacités, les champs, les diagnostics et les déclarations graphiques. Les résultats archivés restent associés à leur propre version.

## Exécution et types

Les populations ordinaires précèdent les grilles ; chaque groupe conserve l'ordre de déclaration. Les agents sont parcourus par identité native croissante ; les réflexes d'un agent sont exécutés dans l'ordre de la source. Les types supportés comprennent int32, float64, bool, point à trois coordonnées, RGBA, références d'agents et listes ordonnées de références dans les profils documentés. Les opérations flottantes conservent les barrières du profil numérique natif. Les facettes numériques min/max suivent la conversion et l'ordre de GAMA ; step est une métadonnée.

Les imports relatifs sont résolus une fois. Les overrides de membres globaux conservent leur emplacement ; les blocs init globaux composent le fichier d'entrée avant les fichiers importés, conformément à la sonde native. L'héritage d'espèces compose les init parent puis enfant et remplace les membres homonymes. Les actions virtuelles doivent avoir une implémentation concrète avant leur appel. Les actions scalaires non récursives sont développées avec paramètres évalués une seule fois. Les références polymorphes restent à implémenter.

Les dépendances des variables imposent leur ordre d'initialisation et de mise à jour, avec diagnostic des cycles. Les facettes `update:` des scalaires, points et RGBA s'exécutent avant les réflexes du scope. Les getters purs `->` de ces types sont évalués à chaque lecture ; `observe_jit(state)` expose leurs valeurs sans modifier l'état de la simulation. Les getters sont en lecture seule, y compris dans les paramètres externes. Les getters scalaires consommant de l’aléa utilisent le protocole non intrusif décrit ci-dessous ; les getters de listes d’agents et les réductions ordonnées qui les lisent sont également couverts dans le profil spatial vérifié.

Les listes globales typées de valeurs int/float/bool/point/RGBA utilisent des constructeurs littéraux immuables et une indexation vérifiée. Les listes globales de références permettent l'ajout, la suppression de la première occurrence et le remplacement ordonné, avec erreur explicite de capacité. Les doublons sont conservés. `shuffle` applique le parcours décroissant natif des listes ; il diffère du shuffle croissant employé par certaines requêtes spatiales. Les noms de couleurs acceptés sont ceux vérifiés contre l'oracle, avec les composantes CSS natives.

## Aléa et initialisation

Le backend implémente les tirages Java et Mersenne, la conversion native des seeds et la consommation conditionnelle des tirages. Les profils acceptés distinguent le reseeding explicite avant ou après initialisation et l'initialisation Mersenne implicite des mondes rectangulaires et des espèces sans skill ou avec le profil moving documenté ci-dessous. Un seed externe fini et non nul est obligatoire pour ce dernier profil.

`rnd` accepte aussi des bornes scalaires calculées de même type. Les intervalles inversés ou non finis produisent une erreur explicite ; ils ne consomment aucun tirage. Les intervalles valides ont été comparés, avec branches conditionnelles, sur dix seeds et cent cycles dans la sonde `dynamic_random_bounds`.

Les créations initiales multiples, effectifs résolus depuis les valeurs globales par défaut, corps init et corps distants sont exécutés dans leur ordre natif. Les positions implicites des agents consomment leurs tirages même lorsqu'elles sont ensuite remplacées. Les grilles minimales rectangulaires utilisent l'initialisation par attribut, les voisinages VN/Moore ordonnés, les shuffles et le comportement toroïdal de GAMA 2025.06.4, y compris sa particularité aux frontières rectangulaires.

## Populations et identités

`compile_model(..., capacities={...})` réserve les tableaux des espèces. `reset(seed=..., parameters={...})` remplace paresseusement les initialisateurs globaux avant application de leurs facettes ; les affectations init restent exécutées. Les démographies externes peuvent varier dans les capacités explicites.

La création au runtime exige une capacité explicite et le profil d'initialisation aléatoire des espèces sans skill. Les identités sont allouées de manière monotone et ne sont pas réutilisées après une mort. La capacité borne donc le nombre total d'identités allouées pendant le run. Les parcours prennent un instantané des identités présentes ; les nouveau-nés de la même espèce ne rejoignent pas le parcours en cours. `do die` masque l'agent et arrête ses effets suivants. Les attributs `myself` des corps distants lisent le créateur. Les asks imbriqués et captures lexicales sont admis dans le profil ordonné vérifié. La naissance/mort depuis un bloc init et les skills implicites à la naissance restent exclus.

Tout dépassement de capacité est une erreur explicite du runtime et exclut la fidélité et la performance. `identities(state)` et les exports de traces préservent les identités natives, indépendamment du compactage des observations.

## Opérations spatiales

Les grilles rectangulaires à voisinages cachés utilisent des agents minimaux ou réguliers. Les dimensions littérales ou résolues depuis les paramètres globaux par défaut, centres et coordonnées de cellules, voisinages persistants en lecture seule, rayon extérieur borné, sélections ordonnées et `accumulate` sont couverts. Les paramètres déterminant les dimensions résolues sont figés dans le module ; les changer exige une recompilation et est refusé par reset.

Le profil moving implicite initialise la position avec deux tirages, puis le heading avec un tirage dans [0,360], speed à 1 et real_speed à 0. `move` accepte des facettes speed/heading pures ; `wander` accepte amplitude/speed pures dans le monde carré. `goto` vers une position ou référence ponctuelle suit le profil continu vérifié, y compris hors du monde ; `move`/`wander` disposent aussi du profil toroïdal validé. Les contraintes personnalisées et déplacements sur graphe restent exclus. Les facettes aléatoires sont refusées tant que leur ordre natif d'évaluation n'est pas vérifié. Les sondes de défauts moving et de grilles régulières passent chacune dix seeds et cent cycles.

`overlapping` couvre actuellement les agents ponctuels dans une cellule de grille minimale, uniquement lorsque l'analyse démontre que le consommateur effectue des sommes int32 locales indépendantes de l'ordre. Les prédicats et incréments sont purs, sans dépendance entre accumulateurs ; les écritures persistantes, l'aléa du corps et les sommes flottantes sont rejetés. Les deux shuffles natifs conservent tous leurs tirages ; leur ordre est éliminé parce qu'il est inobservable dans ce profil. Cette règle réutilisable suffit à la boucle locale du SIR officiel. Le profil batch peut regrouper ces sommes en histogrammes de références de cellules après vérification, pendant la simulation, des centres finis rectilignes et séparés ainsi que de l’alignement des points et de la validité des références. Les multiplicités, l’int32 modulaire et le budget de shuffles sont inchangés ; un garde faux conserve la boucle géométrique. Les interactions Boids sur géométries étendues restent à implémenter. Un second profil couvre maintenant `at_distance` sur collections d’agents ponctuels dans un monde carré non toroïdal : quadtree natif persistant, ordre de parcours, réinsertions et deux shuffles natifs conservés. Les rayons invalides sont signalés. Les requêtes pendant l’initialisation et les changements de forme du monde sont refusés. La sonde ordonnée passe dix seeds et cent cycles ; des sondes de 250 et 1 000 agents vérifient aussi les subdivisions profondes de l’index.

## Sorties et rendu

Les writes globaux init inconditionnels sont évalués et conservés. Les aspects et sorties GUI déclaratives sont archivés séparément dans le manifeste ; les effets de simulation dans ces déclarations sont rejetés. Le backend de simulation n'exécute pas la GUI GAMA. La préservation des tokens graphiques ne constitue pas une implémentation de leur rendu.

Les paramètres GUI simples décrivent les variables réglables ; la configuration du moteur passe par reset. FSM, matrices, diffusion, assets nécessaires à la dynamique et contrôle d'arrêt ne sont pas encore couverts. Une opération inconnue ou sans génération backend entraîne un diagnostic ; aucun tel effet n'est ignoré silencieusement.

## Vérification

La comparaison exige l'égalité des états discrets et de leurs types, longueurs et ordres, les mêmes identités, et `atol=1e-12`, `rtol=1e-10` pour les flottants. Les runners enregistrent les erreurs GAMA, les divergences et les exceptions. Les observations WebSocket convertissent les doubles en chaînes à précision de round-trip, pour éviter l'arrondi du sérialiseur JSON natif.

L'oracle WebSocket recrée une simulation avec le seed demandé et vérifie l'activation sans effet avant les cycles mesurés. Son préchargement supplémentaire le réserve à la fidélité ; il ne fournit pas une baseline de performance. Chaque run officiel reçoit une copie privée des sources et assets afin que les exports du modèle ne modifient pas le corpus figé. Les variantes indépendantes restent non compilées.


Les getters scalaires aléatoires disposent d’une observation non intrusive : le module JAX produit un instantané avec un RNG privé, puis restitue le RNG original. L’oracle GAMA exécute une action d’observation dans le scope de la simulation et restaure le seed et le nombre de mots RNG consommés. Le premier prototype dans le scope de l’expérience échouait ; cet échec reste archivé. La [sonde corrigée](../results/random-getters-ten-seeds-coupled-2026-10-13/summary.json) passe les dix seeds et cent cycles, y compris les tirages ultérieurs. Ce protocole concerne la fidélité, sans mesure de performance. La [sonde de listes spatiales](../results/spatial-getters-ten-seeds-coupled-2026-10-13/summary.json) passe aussi dix seeds et cent cycles : ordre des voisins, réductions calculées, gardes et déplacements concordent. Cette validation de primitives ne constitue pas l’acceptation d’un modèle officiel.


## Extensions pour les sources complètes : fonctions, FSM et diffusion

Les actions typées retournent des scalaires, points, couleurs ou références. Les paramètres sont capturés et évalués une fois dans le contexte appelant ; `return` masque les instructions et tirages suivants. Les fonctions récursives et retours partiels sont rejetés. Les effets restent analysés dans les appels développés, y compris les morts et créations.

Les paramètres `init:` littéraux de l’expérience sélectionnée, avec héritage et overrides, remplacent les initialisateurs globaux avant résolution des effectifs et dimensions. Les overrides externes de `reset` ont priorité. Les facettes GUI ne sont pas des règles de simulation. Les créations `with:(...)` pures remplacent les valeurs initiales des attributs avant l’init ; une position fournie évite les deux tirages de position implicite, conformément à la sonde native.

Les FSM ont exactement un état initial. Les réflexes ordinaires précèdent le corps de l’état courant ; les transitions sont examinées après ce corps, dans l’ordre source, et seule la première vraie est prise. Le nouvel état s’exécute au cycle suivant. Les noms sont préservés dans les observations canoniques. L’héritage de FSM, les enter/exit, états finaux et la mémoire persistante des variables locales restent exclus.

Les casts de grille depuis un point et les voisinages cachés conservent les références ordonnées. En contexte de grille, `distance_to` suit la distance topologique native ; dans un contexte continu, la distance à une cellule utilise sa surface rectangulaire. Une conversion `point(agent)` sur référence vide produit un point local nullable ; son affectation à la position moving est sans effet et n’arrête pas les instructions suivantes. Les points persistants nullable sont refusés. La distance continue à une cellule toroïdale reste exclue. `sort_by` accepte un score numérique pur et conserve les ex aequo ; les comparateurs aléatoires sont rejetés.

Les matrices int/float sont bornées et indexées dans l’ordre natif `(x,y)` ; leurs observations sont aplaties par lignes. Les images immuables et assets sont bundlés avec SHA-256. Les pixels ARGB et casts RGBA suivent le contrat natif. Le redimensionnement `as_matrix(file,{w,h})` s’effectue à la compilation via Java standard ImageIO/AWT, sans classes GAMA ; JDK/version/hash sont enregistrés. Le module émis contient les pixels et ne nécessite ni Java, ni Pillow, ni GAMAX, ni GAMA. Les constructeurs SVG/OBJ/font exclusivement utilisés pour la présentation sont classifiés, archivés et explicitement exclus du backend headless ; un usage dans la dynamique est rejeté.

La diffusion prise en charge est `gradient` par `convolution`, sur un attribut float d’une grille rectangulaire non toroïdale : rayon fixe borné, proportion finie dans `[0,1]`, voisinage Moore ou VN. Une proportion zéro demande la valeur native par défaut un. La requête globale est différée jusqu’après les populations ; la convolution prend le maximum des contributions, conserve la valeur source lorsque le résultat ne l’augmente pas, et ne normalise pas les frontières. Les requêtes multiples pour un même attribut, masques, diffusion toroïdale et autres méthodes sont rejetés.

Le profil numérique utilise les options JIT natives aussi pour l’initialisation lorsqu’une division constante non binaire doit être conservée. Les états discrets restent comparés exactement, les flottants avec `atol=1e-12`, `rtol=1e-10`. Le profil ne prouve ni l’équivalence de tout GAML, ni celle de tous les cas numériques particuliers. Les sauvegardes texte relatives sont des événements explicites ; le format natif n’ajoute pas automatiquement de fin de ligne. Le contrôle `pause` expose un indicateur interne ; les campagnes à horizon fixé continuent explicitement les pas et ne prouvent pas un pilote autonome reproduisant tous les arrêts GUI.


## Optimisation facultative

`compile_model` accepte `optimization="none"` (défaut) ou `"safe"`. Les passes interviennent sur l’IR typée et sont décrites dans le manifeste. Elles conservent le contrat numérique et l’ordre natif ; elles ne constituent pas une certification automatique de fidélité. Voir le [contrat de l’optimiseur](optimizer.md) pour les conditions de la première passe et les preuves différentielles.

### Réutilisation de `each` après un getter itérateur

GAMA 2025.06.4 réaffecte le binding natif de `each` dans les lambdas de conteneurs, sans le restaurer au retour. Une expression qui lit de nouveau le `each` appelant après un getter contenant un itérateur n'est pas encore traduite avec cette sémantique dynamique. Le compilateur la rejette désormais avec `GAMAX_EACH_SCOPE`, avant émission. Le [probe initial](../results/optimizer-pure-getter-native-probe/summary.json) et le [contrôle de référence](../results/optimizer-pure-getter-reference-check/summary.json) conservent la divergence identique des profils optimisé et non vectorisé ; le [probe avec getter booléen autonome](../results/optimizer-pure-getter-native-probe-v2/summary.json) couvre une construction distincte et passe 159 137 valeurs et identités. Aucun modèle officiel n'a été adapté pour contourner cette restriction.


### Sauvegardes et conversion scalaire des sorties

Les profils optimisés émettent une sérialisation scalaire compatible avec la notation du JDK 21 vérifiée dans les tests : booléens, entiers, notation scientifique des flottants, subnormaux et zéros signés. La référence historique `none` conserve son émission byte-identique ; ses conversions Python ne constituent pas une preuve d'identité des fichiers texte avec Java. Le runner compare toujours les fichiers exacts et garde les rejets.

Une sauvegarde est collectée seulement quand la génération de son statement avance. Le payload reste conservé dans l'état lorsque sa garde devient fausse, mais il n'est pas réémis. Le format d'état actuel réserve un seul payload par statement ; plusieurs émissions du même statement pendant une initialisation ou un cycle ne sont pas reconstructibles. `run` et le runner de performance signalent explicitement ce cas au lieu de perdre silencieusement les payloads. La prise en charge générale de plusieurs sauvegardes par statement/cycle reste à implémenter.
