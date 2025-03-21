import streamlit as st

DEBUGGING = st.secrets.get("DEBUGGING", False)


def show_schema_in_sidebar():
    """
    Displays, in the sidebar, the database tables, columns,
    foreign keys, exact-match format info (years, schools, types),
    the official 52 questions (with question IDs, texts, type IDs),
    plus additional info on Répartition des écoles and question distribution.
    """

    if DEBUGGING:

        st.sidebar.title("Database Schema and Values")  # --- TABLES ---
        with st.sidebar.expander("Table: schools"):
            st.write("**Columns**:")
            st.write("- school_id | Type: INTEGER")
            st.write("- school_name | Type: VARCHAR(100)")

        with st.sidebar.expander("Table: questionnaires"):
            st.write("**Columns**:")
            st.write("- questionnaire_id              | Type: VARCHAR(100)")
            st.write("- school_id                     | Type: INTEGER")
            st.write("- type_id                       | Type: INTEGER")
            st.write("- year                          | Type: INTEGER")
            st.write("- total_graduates               | Type: INTEGER")
            st.write("- questionnaire_completed       | Type: INTEGER")
            st.write("- percent_answered              | Type: DOUBLE PRECISION")
            st.write("- female_count                  | Type: INTEGER")
            st.write("- male_count                    | Type: INTEGER")
            st.write("**Foreign Keys**:")
            st.write("- questionnaires.type_id -> types.type_id")
            st.write("- questionnaires.school_id -> schools.school_id")

        with st.sidebar.expander("Table: types"):
            st.write("**Columns**:")
            st.write("- type_id | Type: INTEGER")
            st.write("- type    | Type: VARCHAR(100)")

        with st.sidebar.expander("Table: questions"):
            st.write("**Columns**:")
            st.write("- question_id   | Type: INTEGER")
            st.write("- question_text | Type: TEXT")
            st.write("- type_id       | Type: INTEGER")
            st.write("**Foreign Key**:")
            st.write("- questions.type_id -> types.type_id")

        with st.sidebar.expander("Table: responses"):
            st.write("**Columns**:")
            st.write("- response_id                  | Type: INTEGER")
            st.write("- questionnaire_id            | Type: VARCHAR(50)")
            st.write("- question_id                 | Type: INTEGER")
            st.write("- gender                      | Type: VARCHAR(1)")
            st.write("- response_option             | Type: VARCHAR(100)")
            st.write("- summed_students_responses | Type: INTEGER")
            st.write("**Foreign Keys**:")
            st.write("- responses.question_id -> questions.question_id")
            st.write("- responses.questionnaire_id -> questionnaires.questionnaire_id")

        # --- VALUES EXACT MATCH FORMAT ---
        st.sidebar.header("Values Exact Match Format")

        with st.sidebar.expander("Years"):
            st.write("From 2004 to 2019")

        with st.sidebar.expander("school_id : School Names"):
            st.write(
                """
    1: Aux quatre vents  
    2: Centre La Fontaine  
    3: Secondaire Népisiguit  
    4: Louis-Mailloux  
    5: Marie-Esther  
    6: Roland-Pépin  
    7: W.-A.-Losier
            """
            )

        with st.sidebar.expander("type_id : type"):
            st.write(
                """
    1: Questions Générales  
    2: SD-Renseignements Socio-Démographiques  
    3: ED- Éducation PostSecondaire  
    4: MT- Marché du travail  
    5: RE-Attente d'emploi / Recherche d'emploi / Sans emploi
            """
            )
        with st.sidebar.expander("Questions (Total: 52)"):
            st.write(
                """
                    - **QG:** 17 | 16-> 32  
                    - **SD:** 8 | 45->52
                    - **ED:** 10 | 1->10
                    - **MT:** 5 | 11->15
                    - **RE:** 12 | 33->44
                    """
            )
        with st.sidebar.expander(
            "Écoles (2004-2019) : Années non-participation | Années participation au questionnaire type 2 "
        ):
            st.write(
                """
                - **QPME (Marie-Esther):** 0 | 8 (2004–2008, 2010–2012)
                - **QCSCLF (Centre Lafontaine):** 1 (2013) | 8 (2004–2008, 2010–2012)
                - **QWAL (W. A.-Losier):** 0 | 8 (2004–2008, 2010–2012)
                - **QPRP (Roland-Pépin):** (2004 - 2013) | 0
                - **QAQV (Aux quatre vents):** (2004 - 2013) | 0
                - **QPLM (Louis-Mailloux):** 0 | 8 (2004–2008, 2010–2012)
                - **QESN (Secondaire Népisiguit):** (2004 - 2013) | 0
                """
            )


def show_questions_in_sidebar():

    st.sidebar.subheader("Questions aux étudiants")

    # Q1
    with st.sidebar.expander(
        "Q1: Dans quelle province canadienne ou ailleurs avez-vous étudié ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write(
            """[
    '(Angleterre)', '(Québec)', '(Manitoba)', '(Non répondu)',
    '(Alberta)', '(Saskatchewan)', '(Colombie-Britannique)',
    '(Ontario)', '(France)', '(Île-du-Prince-Édouard)',
    '(Nouveau-Brunswick)', '(Nouvelle-Écosse)', '(Terre-Neuve)'
    ]"""
        )

    # Q2
    with st.sidebar.expander(
        "Q2: Durant vos études quelles étaient vos sources de financement ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write(
            """[
    'Bourse étudiants', 'Prêts étudiant', 'Économie personnelle', 'Travail à temps partiel',
    "Plan enregistré d'épargne-études", 'Autre', 'Travail à temps plein',
    'Prêts/bourse étudiant', 'Bourse étudiante', "Programme d'aide fédéral",
    'Travaillle à temps plein', "Plan enregistré d'épargnes-études",
    "Programme d'aide provincial", 'Travaille à temps partiel', 'Non répondu', 'Contribution des parents'
    ]"""
        )

    # Q3
    with st.sidebar.expander(
        "Q3: Seriez-vous intéressé/e à faire un retour aux études ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write(
            "['Non', 'Ne sais pas', 'Présentement aux études', 'Non répondu', 'Oui']"
        )

    # Q4
    with st.sidebar.expander(
        "Q4: Existe-t-il des obstacles qui vous empêchent de poursuivre des études ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write("['Non répondu', 'Non', 'Oui']")

    # Q5
    with st.sidebar.expander(
        "Q5: Seriez-vous intéressé/e à recevoir de l'information pour vous aider dans la poursuite de vos études ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write("['Non répondu', 'Non', 'Oui']")

    # Q6
    with st.sidebar.expander(
        "Q6: Dans quel type d'institution avez-vous étudié ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write("['Universitaire', 'Non répondu', 'Collégial', 'École privée']")

    # Q7
    with st.sidebar.expander(
        "Q7: Quel était votre régime d'études ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write("['Temps plein', 'Non répondu', 'Temps partiel']")

    # Q8
    with st.sidebar.expander(
        "Q8: Quel était le niveau de votre programme d'études ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write(
            """[
    'Troisième cycle ( doctorat )',
    'Deuxième cycle ( maîtrise )',
    'Premier cycle universitaire( bacc., certificat)',
    'Diplôme', 'Certificat', 'Non repondu',
    'Cours préparatoire ( 13e année )',
    'Sciences de la santé',
    'Non répondu'
    ]"""
        )

    # Q9
    with st.sidebar.expander(
        "Q9: Dans quel domaine d'études étiez-vous inscrit/e ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write(
            """[
    'Mathématiques, informatique et sciences physiques',
    'Professions de la santé et technologies connexes',
    'Commerce, gestion et administration des affaires',
    'Beaux-arts et arts appliqués',
    'Lettres, sciences humaines et disciplines connexes',
    'Enseignement, loisirs et orientation',
    'Sciences sociales et disciplines connexes',
    'Non repondu',
    'Génie et sciences appliquées',
    'Sans spécialisation',
    'Non répondu',
    'Techniques et métiers des sciences appliquées'
    ]"""
        )

    # Q10
    with st.sidebar.expander(
        "Q10: Avez-vous complété vos études ? (Type ID: 3, Type: ED- Éducation PostSecondaire)"
    ):
        st.write("['Non répondu', 'Non', 'Oui']")

    # Q11
    with st.sidebar.expander(
        "Q11: Dans quelle province canadienne ou autre pays travaillez-vous ? (Type ID: 4, Type: MT- Marché du travail)"
    ):
        st.write(
            """[
    '(Angleterre)', '(Québec)', '(Manitoba)', '(Non répondu)', '(Territoires-du-Nord-Ouest)',
    '(Alberta)', '(Saskatchewan)', '(Colombie-Britannique)', '(Nunavut)', '(Ontario)',
    '(France)', '(Île-du-Prince-Édouard)', '(Nouveau-Brunswick)', '(Suisse)', '(Nouvelle-Écosse)',
    '(Terre-Neuve)'
    ]"""
        )

    # Q12
    with st.sidebar.expander(
        "Q12: Envisagez-vous de changer d'emploi dans la prochaine année ? (Type ID: 4, Type: MT- Marché du travail)"
    ):
        st.write("['Non répondu', 'Non', 'Oui']")

    # Q13
    with st.sidebar.expander(
        "Q13: Quelle est la principale raison pour laquelle vous envisagez de changer d'emploi ? (Type ID: 4, Type: MT- Marché du travail)"
    ):
        st.write(
            """[
    "Je veux travailler plus près de chez moi",
    "J'ai besoin d'un changement",
    "Mon travail actuel ne correspond pas à mon expérience et à ma formation",
    "Je travaille à temps plein et je préférerais travailler à temps partiel",
    "Je recherche la sécurité d'emploi",
    "Je désire retourner aux études"
    ]"""
        )

    # Q14
    with st.sidebar.expander(
        "Q14: Dans quelle région de la province travaillez-vous ? (Type ID: 4, Type: MT- Marché du travail)"
    ):
        st.write(
            """[
    'Nord-Est', 'Ouest', 'Nord', 'Nord-Ouest', 'Sud', 'Autre',
    'Centre', 'Est', 'Non Répondu', 'Sud-Est', 'Non répondu',
    'Sud-Ouest', 'non répondu'
    ]"""
        )

    # Q15
    with st.sidebar.expander(
        "Q15: Dans quelle catégorie de profession ou métier travaillez-vous ? (Type ID: 4, Type: MT- Marché du travail)"
    ):
        st.write(
            """[
    'Commerce de détail',
    'Arts, spectacles et loisirs',
    'Fabrication',
    'Services professionnels, scientifiques et technique',
    'Agriculture, forestierie, pêche et chasse',
    "Services admistratifs, services de soutien, services de gestion de déchets et d'assainissement",
    'Finance et assurances ',
    'Commerce en gros',
    "Insdustrie de l'information - industrie culturelle",
    'Transport et entreposage',
    'Non répondu',
    'Services publiques',
    'Hébergement et services de restauration',
    "Gestion de sociétés et d'entreprises",
    'Autre services (sauf administration publique)',
    "Services d'enseignement",
    'Soins de santé et assistance sociale',
    'Extraction minière, extraction de pétrole et de gaz ',
    'Services immobiliers, service de location',
    'Construction',
    'Administration publiques'
    ]"""
        )

    # Q16
    with st.sidebar.expander(
        "Q16: Indiquez les deux cours que vous avez les plus appréciés durant votre secondaire (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            """[
    '46. Sciences familiales', "45. Sc. De l'environnement", '52. Technologie', '18. Droit',
    '22. Éducation coopérative', '36. IPEJ', '49. Sciences sociales', '10. Biologie', '29. Français',
    '39. Mécanique des moteurs', "35. Intro. Science de l'info.", '48. Sciences humaines',
    '50. Systèmes mécaniques', '14. Chimie', '42. Physique', '47. Sciences générales', '37. Leadership',
    "54. Trait. De l'information", '3. Alimentation et nutrition', '44. Saisie sur clavier', '12. Braille Nameth',
    '20. Éduc. Du consommateur', '5. Anglais', '53. Textiles et habillement', '9. Atelier de métaux', '2. Agriculture',
    '40. Menuiserie', '19. Économie', '6. Art dramatique', '33. Informatique appliqué', '1. Act.Quot.(handicap vis.)',
    '43. Relations familiales', '11. Braille abrégé', '13. Charpenterie', '30. Géographie', '28. Formation personnelle',
    '17. Développement humain', '15. Comptabilité', '27. Espagnol', '51. Techno. (handicap visuel)', '26. Entreprenariat',
    '7. Arts Industriels', '34. Initiation au travail', '41. Musique', '25. Électronique', '24. Électricité', '31. Histoire',
    '32. Informatique', '38. Mathématiques', '8. Arts visuels', '16. Dessin industriel', '4. Allemand',
    '21. Éducation aux valeur', '23. Éducation physique'
    ]"""
        )

    # Q17
    with st.sidebar.expander(
        "Q17: Sur le plan de vos relations avec les autres élèves, êtes-vous satisfait/e de votre séjour au secondaire ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            "['Satisfait/e', 'Très insatisfait/e', 'Insatisfait/e', 'Non répondu', 'Très satisfait/e']"
        )

    # Q18
    with st.sidebar.expander(
        "Q18: Sur le plan de vos relations avec le personnel de l'école, êtes-vous satisfait/e de votre séjour au secondaire ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            "['Satisfait/e', 'Très insatisfait/e', 'Insatisfait/e', 'Non répondu', 'Très satisfait/e']"
        )

    # Q19
    with st.sidebar.expander(
        "Q19: Que prévoyez-vous faire en septembre prochain ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            """[
    'Marché du travail', 'Année sabbatique ', 'Programmes jeunesse', 'Ligne de hockey',
    'Formation professionnelle ', 'Retours aux études secondaires', 'Voyage pour Missionnaire ',
    'Cégep ', 'Études collégiales ', 'Ne sais pas ', 'Non répondu', 'Voyager ', 'Études universitaires',
    'Écoles des pêches ', 'Indécis du domaine souhaiter ', 'Programme jeunesse',
    'Cours se donne seulement aux 2 ans ', 'Pas assez de bonnes notes', 'Cours par correspondance ',
    'Armée'
    ]"""
        )

    # Q20
    with st.sidebar.expander(
        "Q20: Indiquez dans quel domaine vous avez l'intention de poursuivre vos études postsecondaires ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            """[
    'Mathématiques, informatique et sciences physiques',
    'Sciences agricoles et biologiques et services de la nutrition et de l&rsquo;alimentation',
    'Professions de la santé et technologies connexes',
    'Commerce, gestion et administration des affaires',
    'Beaux-arts et arts appliqués',
    'Lettres, sciences humaines et disciplines connexes',
    'Enseignement, loisirs et orientation',
    'Sciences sociales et disciplines connexes',
    'Génie et sciences appliquées',
    'Sans spécialisation',
    'Non répondu',
    'Techniques et métiers des sciences appliquées'
    ]"""
        )

    # Q21
    with st.sidebar.expander(
        "Q21: Est-ce que les services d'orientation que vous avez utilisés ont répondu à vos besoins ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            "['Pas utilisés', 'Peu', 'Bien', 'Pas du tout', 'Très bien', 'Non répondu']"
        )

    # Q22
    with st.sidebar.expander(
        "Q22: À la fin de vos études postsecondaires, si vous aviez une offre d'emploi dans la Péninsule acadienne, est-ce que vous l'accepteriez ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write("['Non', 'Non répondu', 'Je ne sais pas', 'Oui']")

    # Q23
    with st.sidebar.expander(
        "Q23: Lorsque vous aurez terminé vos études postsecondaires, indiquez où vous prévoyez travailler ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            """[
    '4. Charlotte (N.-B.)', '19. Nouvelle-Écosse', '20. Ontario', '26. Yukon', '35. Canada',
    '17. IPE', '31. Autre', '11. Restigouche (N.-B.)', '24. Alberta', '27. Territoire du Nord-Ouest',
    '16. York (N.-B.)', '33. Autre(s) province(s)', '23. Saskatchewan', '6. Kent (N.-B.)',
    '1. Dans la Péninsule acadienne', '32. Nouveau-Brunswick', '12. Sunbury (N.-B.)', '2. Albert (N.-B.)',
    '10. Queens (N.-B.)', '21. Québec', '18. Terre-Neuve', '3. Carleton (N.-B.)', '9. Northumberland (N.-B.)',
    '7. Kings (N.-B.)', '30. Angleterre', '29. France', '8. Madawaska (N.-B.)', '14. Victoria (N.-B.)',
    '28. État-Unis', '13. Saint-Jean (N.-B.)', '22. Manitoba', '5. Gloucester (N.-B.)', '15 Westorland (N.-B.)',
    '34. Autre(s) pays', '25. Colombie-Britanique'
    ]"""
        )

    # Q24
    with st.sidebar.expander(
        "Q24: Indiquez le nom de l'institution que vous pensez fréquenter l'an prochain ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            """[
    'Université de Sherbrooke', 'University Mount Allison', 'Notre-Dame-de-Foy', 'Université Western',
    'Recording Arts Canada', 'Oulton College', 'CITÉ COLLÉGIALE', 'Eastern College', 'MEDES Institutes',
    'SALON CAROUSSEL', 'Université de Moncton - campus Moncton', 'NDI', 'Université Laval',
    'Académie de Police (Régina)', 'CCNB (PA)', 'UNIVERSITE LAVAL', 'NBCC Miramichi Campus', 'CCNB MIRAMICHI',
    'Nav Canada', 'College de la Garde Cotière Canadienne', 'Université St-Francis Xavier', 'ESSENCE INSTITUT',
    'UNIVERSITÉ DE MONTRÉAL', 'École de coiffure Rose-Hélène', 'Autre', 'CCNB Moncton', 'OULTON COLLEGE',
    '2. Bourses étudiantes', '4. Économie personnelle', 'ACADEMY OF LEARNING MIRAMICHI', 'Brenda dog grooming',
    'Académie paramédicale de lAtlantique', 'Universdad de Boenos', 'NAIT', '1. Prêt étudiant',
    "UNIVERSITÉ D'OTTAWA", 'UNIVERSITÉ STE-ANNE', 'CCNB (Bathurst)', 'ÉCOLE DES PÊCHES', 'THE ART INSTITUTE OF TORONTO',
    '6. Autre', 'MEMORIAL UNIVERSITY', 'St-Thomas University', 'Cégep', 'Université d Acadia', 'COLLÈGE LASALLE',
    "École nationale opération de machinerie lourde", 'Université de Guelph', 'Mount Allison', 'Académie La Coupe Plus',
    'CCNB (Campbelton)', 'Université de Moncton - campus Edmunston', 'Université McGill', 'COLLÈGE BORÉAL',
    'College de technologie forestiere des Maritimes', 'COLLÈGE MILITAIRE ROYAL DU CANADA', 'CCNB Campbellton-Cours distance',
    'Centre de formation de la Haute-Gaspésie', 'Institut humanitaire', 'University of Mobile',
    'Le Collège Atlantic de Massage thérapeutique', 'centre de formation professionnelle , Trois Rivières. qc',
    'CRIF', 'Université de Montréal', 'CCNB (Edmunston)', 'CCNB (Autre)', 'CITE COLLÉGIALE',
    "Dalhousie  University", "Université St Mary's", "ECOLE DES MÉTIERS DE L'INDUSTRIE DE LA CONSTRUCTION",
    "INST. TECH.D'AGROALIMENTAIRE", "Université de l'Ile du Prince Edouard", '3. Contribution des parents',
    'Campus Notre Dame de Foy', 'NBCC MONCTON', 'UCAM', 'Université de Moncton - campus Bathurst',
    "Université du Nouveau-Brunswick", 'Université de Concardia', 'CCNB Fredericton',
    'Collège Acadie, Université Ste-Anne', 'Amor Esthétique', 'Université Hearst',
    'Maritime College of forest technololgy', 'ÉCOLE DE COIFFURE DE LA PÉNINSULE', 'UNIVERSITÉ LAURENTIENNE',
    'UNIVERSITÉ DE MONCTON(CAMPUS DE SHIPPAGAN ET MONCTON)', "École National du Théatre du Canada", 'Da Vinci College',
    'Cégep de Ste Foy', 'Université de Moncton - campus Shippagan', 'Université de Toronto, campus St-George',
    'Institut Maritime du Québec', 'JON RAYMOND', 'MEDES', 'Université du Québec',
    'Collège Atlantique de massages thérapeutique', 'CEGEP DE LAPOCATIÈRE', 'Amoura Esthetic',
    '5. Travail à temps partiel', 'Non répondu', 'CCNB (Dieppe)', 'St Thomas University',
    'Université du Canadienne', "BRENDA'S ACADEMY OF PROFESSIONNAL DOG GROOMING", 'Holland College',
    'Université de Dalhousie', 'NBCC', 'Université libre de Bruxelles', 'Medavie HealtEd',
    'Complexe Capital Hélicoptère', 'NBCC Monction', 'Bathurst Hair academy', 'Collège Merici', 'UNB',
    'Cégep de Rimouski'
    ]"""
        )

    # Q25
    with st.sidebar.expander(
        "Q25: Durant vos études postsecondaires, quelles seront vos sources de financement ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            """[
    '3. Contribution des parents', '2. Bourses étudiantes', '4. Économie personnelle',
    '1. Prêt étudiant', '5. Travail à temps partiel', '6. Autre'
    ]"""
        )

    # Q26
    with st.sidebar.expander(
        "Q26: Quel moyen de transport local pensez-vous utiliser le plus régulièrement pendant votre première année d'études ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            """[
    '11. Covoiturage', '10. Taxi', '7. Voiture frère/sœur',
    '3. Voiture de mes parents', '1. À pied', '8. Transport en commun',
    '2. Ma propre voiture/motocyclette', '6. Métro', '5. Autobus', '12. Autre',
    '9. Bicyclette', '6. Autre', "4. Voiture d'un ami/e ami/e"
    ]"""
        )

    # Q27
    with st.sidebar.expander(
        "Q27: Quelle est la raison pour laquelle vous ne prévoyez pas poursuivre des études postsecondaire ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            """[
    'Accès au marché du travail', 'Peur de ne pas réussir', 'Cours non disponible',
    'Aucune idée du domaine à choisir', 'Manque d’intérêt pour les études/formation',
    'Je travaille ou prévois travailler', 'Autre', 'Prendre un an', 'Armée', 'Problèmes monétaires',
    'Non répondu'
    ]"""
        )

    # Q28
    with st.sidebar.expander(
        "Q28: Depuis septembre dernier, avez-vous occupé un emploi rénuméré ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            """[
    '5. Autres', '1. Non', "4. à l'occasion (ex. 1-2 fois par semaine)",
    '2. Temps partiel fin de semaine', '3. Temps partiel à la semaine'
    ]"""
        )

    # Q29
    with st.sidebar.expander(
        "Q29: Est-ce que le fait de terminer votre 12e année est important pour vous ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            "['Important', 'Peu important', 'Pas du tout important', 'Très important', 'Non répondu']"
        )

    # Q30
    with st.sidebar.expander(
        "Q30: En général, quels sont vos résultats scolaires en 10e 11e et 12e année ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            "['de 81 % à 94 %', 'de 56 % à 65 %', 'de 95 % à 100 %', 'moins de 55 %', 'de 66 % à 80 %', 'Non répondu']"
        )

    # Q31
    with st.sidebar.expander(
        "Q31: En général, vos cours sont de quel niveau scolaire ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            "['Adaptation scolaire', 'PCE', 'Régulier', 'Compétences essentielles', 'Choix', 'Non répondu', 'Modifié']"
        )

    # Q32
    with st.sidebar.expander(
        "Q32: En général, êtes-vous satisfait/e de vos résultats scolaires ? (Type ID: 1, Type: Questions Générales)"
    ):
        st.write(
            "['Satisfait/e', 'Très insatisfait/e', 'Insatisfait/e', 'Non répondu', 'Très satisfait/e']"
        )

    # Q33
    with st.sidebar.expander(
        "Q33: Présentement êtes-vous en attente ou à la recherche d'emploi ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write(
            """[
    "Recherche d'emploi", 'Non', "Attente d'emploi", 'Non répondu', 'Oui'
    ]"""
        )

    # Q34
    with st.sidebar.expander(
        "Q34: Au cours des 2 dernières années, combien d'employeurs avez-vous eus ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write("['Un', 'Aucun', 'Plus de deux', 'Non répondu', 'Deux']")

    # Q35
    with st.sidebar.expander(
        "Q35: Au cours des deux dernières années, combien de mois d'expérience de travail avez-vous accumulés ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write(
            "['Moins de 6 mois', '12 à 18 mois', '18 à 24 mois', '6 à 12 mois', 'Non répondu']"
        )

    # Q36
    with st.sidebar.expander(
        "Q36: Depuis combien de temps êtes-vous sans emploi ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write(
            "['Moins de 6 mois', 'Plus de 2 ans', '1 à 2 ans', '6 à 12 mois', 'Non répondu']"
        )

    # Q37
    with st.sidebar.expander(
        "Q37: Quelle est la principale raison pour laquelle vous n'avez pas travaillé durant cette période ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write(
            """[
    "Attente d'un rappel au travail après mise à pied", 'Invalidité', 'Accidenté',
    "Rémunération insuffisante pour quitter l'A-S ou A-E", 'Changement de propriétaire',
    'Chômage / travail saisonnier', 'Déménagement', 'Pas de voiture', 'Conflit de travail',
    'Incapacité de trouver un emploi lié à ma formation', 'Accouchement', "Attente du début d'un emploi",
    'Chômage de quelques semaines', 'Retour aux études', 'Responsabilité personnelles ou familiales',
    'Non répondu', 'Maladie'
    ]"""
        )

    # Q38
    with st.sidebar.expander(
        "Q38: Seriez-vous intéressé/e par une expérience de travail dans votre communauté ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write("['Non répondu', 'Non', 'Oui']")

    # Q39
    with st.sidebar.expander(
        "Q39: Seriez-vous intéressé à recevoir des l'information concernant une expérience de travail dans votre communauté ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write("['Non', 'Non répondu', 'Oui']")

    # Q40
    with st.sidebar.expander(
        "Q40: Êtes-vous disponible pour travailler maintenant ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write("['Non', 'Non répondu', 'Oui']")

    # Q41
    with st.sidebar.expander(
        "Q41: Quel type d'emploi recherchez-vous ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write(
            """[
    'Commerce de détail', 'Arts, spectacles et loisirs', 'Fabrication',
    'Services professionnels, scientifiques et technique',
    'Agriculture, forestierie, pêche et chasse',
    "Services admistratifs, services de soutien, services de gestion de déchets et d'assainissement",
    'Finance et assurances ', 'Commerce en gros',
    "Insdustrie de l'information - industrie culturelle",
    'Transport et entreposage', 'Non répondu', 'Services publiques',
    'Hébergement et services de restauration', "Gestion de sociétés et d'entreprises",
    'Autre services (sauf administration publique)', "Services d'enseignement",
    'Soins de santé et assistance sociale', 'Extraction minière, extraction de pétrole et de gaz ',
    'Construction', 'Administration publiques'
    ]"""
        )

    # Q42
    with st.sidebar.expander(
        "Q42: Dans quelle province canadienne ou autre pays aimeriez-vous travailler ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write(
            "['Alberta', 'Saskatchewan', 'Autre', 'Colombie-Britanique', 'Ontario', 'Terre-Neuve', 'Nouveau-Brunswick', 'Québec', 'Non répondu', 'non répondu']"
        )

    # Q43
    with st.sidebar.expander(
        "Q43: Dans quelle région de la province aimeriez-vous travailler ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write(
            "['Nord-Est', 'Saskatchewan', 'Nord', 'Sud-ESt', 'Nord-Ouest', 'Sud', 'Colombie-Britannique', 'Autre', 'Terre-Neuve', 'Nouveau-Brunswick', 'Sud-Est', 'Non répondu', 'Sud-Ouest']"
        )

    # Q44
    with st.sidebar.expander(
        "Q44: Avez-vous eu un emploi au cours des 2 dernières années ? (Type ID: 5, Type: RE-Attente d'emploi / Recherche d'emploi / Sans emploi)"
    ):
        st.write("['Non répondu', 'Non', 'Oui']")

    # Q45
    with st.sidebar.expander(
        "Q45: Depuis combien d'années demeurez-vous à l'extérieur de la Péninsule acadienne ? (Type ID: 2, Type: SD-Renseignements Socio-Démographiques)"
    ):
        st.write(
            "['moins d\\'un an', 'Deux ans', 'Un an', 'Trois ans', 'Plus de quatre', 'Non répondu', 'Quatre ans']"
        )

    # Q46
    with st.sidebar.expander(
        "Q46: Pour quelle raison avez-vous quitté la Péninsule acadienne ? (Type ID: 2, Type: SD-Renseignements Socio-Démographiques)"
    ):
        st.write("['Travail', 'Non répondu', 'Étude', 'Autre']")

    # Q47
    with st.sidebar.expander(
        "Q47: Avez-vous l'intention de vous établir dans la Péninsule acadienne au cours des cinq prochaines années ? (Type ID: 2, Type: SD-Renseignements Socio-Démographiques)"
    ):
        st.write("['Non répondu', 'Non', 'Ne sais pas', 'Oui']")

    # Q48
    with st.sidebar.expander(
        "Q48: Quel est votre niveau d'éducation le plus élevé ? (Type ID: 2, Type: SD-Renseignements Socio-Démographiques)"
    ):
        st.write(
            """[
    'Formation personnelle', 'Secondaire (12 ième année)', 'Secondaire', 'Secondaire non-complété',
    'DEG non-complété', 'Universitaire (baccalauréat)', 'non répondu', 'Formation professionnelle',
    'Universitaire (Doctorat)', 'Non répondu', 'Collégial', 'Universitaire (Maîtrise)',
    'DESPA complété', 'DESPA non complété', 'Universitaire (Baccalauréat)', 'DESPA Complété',
    'DEG complété', 'Secondaire (12 ième complété)', 'Autre'
    ]"""
        )

    # Q49
    with st.sidebar.expander(
        "Q49: Que consirérez-vous comme étant votre principale occupation en ce moment ? (Type ID: 2, Type: SD-Renseignements Socio-Démographiques)"
    ):
        st.write(
            """[
    "Recherche d'emploi / sans emploi", 'Termine mes études secondaires', 'Études universitaires',
    'Formation professionnelle ', 'Retours aux études secondaires', "Recherche d'emploi",
    "Recherche d'emploi / Sans emploi", 'Autre', 'Marché du travail', 'Armée', 'Programme jeunesse',
    'Non répondu', 'Termine mes études postsecondaires', 'non répondu'
    ]"""
        )

    # Q50
    with st.sidebar.expander(
        "Q50: Quel est votre état civil présentement ? (Type ID: 2, Type: SD-Renseignements Socio-Démographiques)"
    ):
        st.write(
            "['Divorcé/e', 'Veuf / veuve', 'Célibataire', 'Marié/e', 'Conjoint/e de fait', 'Séparé/e', 'Non répondu']"
        )

    # Q51
    with st.sidebar.expander(
        "Q51: De quelle localité de la Péninsule acadienne êtes-vous originaire ? (Type ID: 2, Type: SD-Renseignements Socio-Démographiques)"
    ):
        st.write(
            """A very long list of localities (over 300 entries), including:
    'AIRDRIE', 'ROBICHAUD SETTLEMENT', 'POINTE-A-BOULEAU', 'ANSE-BLEUE', 'Granby', ... 'BELLEDUNE', ... 'BATHURST', ...
    'NON RÉPONDU', 'Quispamsis', ... 'MONCTON', ... 'BELLEDUNE', 'SAINT-SAUVEUR', ... etc.
    (Refer to original data for the complete list.)
    """
        )

    # Q52
    with st.sidebar.expander(
        "Q52: Où demeurez-vous présentement ? (Type ID: 2, Type: SD-Renseignements Socio-Démographiques)"
    ):
        st.write(
            "['Péninsule acadienne', 'Ailleurs au N.-B.', 'Ailleurs au Canada', 'Dans un autre pays', 'Non répondu']"
        )
