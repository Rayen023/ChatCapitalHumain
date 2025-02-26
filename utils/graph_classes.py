from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import streamlit as st
# Initialize the LLM
MODEL_CONFIG = {
    #"model_name": "anthropic/claude-3.5-sonnet:beta",
    #"model_name": "google/gemini-2.0-pro-exp-02-05:free",
    #"model_name": "openai/gpt-4o-mini",
    #"model_name": "google/gemini-2.0-flash-001",
    # "model_name": "anthropic/claude-3.7-sonnet",
    "model_name": "openai/o3-mini",
    #"model_name": "openai/o3-mini-high",
    "temperature": 0,
    "max_tokens": 8096,
    "timeout": None,
    "max_retries": 2,
    "streaming": True,
}
llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
    openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
    **MODEL_CONFIG,
)
# Define our structured data models
class QueryProposal(BaseModel):
    tables: List[str] = Field(
        description="The tables in the database that would be used to answer the query",
    )
    columns: List[str] = Field(
        description="The columns from the tables that would be relevant for the query",
    )
    fields: List[str] = Field(
        description="The specific fields or values to filter or select in the query",
    )
    explanation: str = Field(
        description="Explanation of how these tables, columns, and fields would be used to answer the query",
    )

class AnalysisResult(BaseModel):
    is_answerable: bool = Field(
        description="Whether the query can be answered with the available database",
    )
    explanation: str = Field(
        description="Explanation of why the query is or is not answerable",
    )
    query_proposal: Optional[QueryProposal] = Field(
        description="Proposal for query structure if answerable",
        default=None
    )

class DatabaseQueryState(TypedDict):
    user_request: str  # The original user query
    analysis_result: Optional[AnalysisResult]  # Result of query analysis
    human_analyst_feedback: Optional[str]  # Feedback from human expert
    final_query_instructions: Optional[str]  # Final instructions for SQL agent

# System prompt for query analysis
query_analysis_instructions = """You are a database query assistant with access to questionnaire response data.

Database Overview:
- The database contains data for each year and questionnaire.
- For every year, each questionnaire includes multiple questions, each offering several answer options.
- For each answer option, the total number of male and female respondents is recorded.
- Important: Data is aggregated at the school level only, not individual students.

Your task is to analyze the user's request and determine:
1. Is the request answerable with the available data?
2. If NOT answerable, explain why (especially if it tries to link multiple questions). 
3. If answerable, suggest the tables, columns, and fields needed for the query.

Remember: It is IMPOSSIBLE to answer queries that attempt to link responses across multiple questions.

Examples:
- Answerable: "What is the total number of male and female responses for question X in school Y for the year 2015?"
- NOT answerable: "How many students who answered question A also answered question B?"
---

Note : For greetings, general non related questions on different subjects, also set to not answerable but in the explanation field answer it normally in a conversational polite manner as a helpful general chatbot.

**1. Database Overview**  
- **Structure:**  
  - The database contains data for each year and questionnaire.  
  - For every year, each questionnaire includes multiple questions, each offering several answer options.  
  - For each answer option, the total number of male and female respondents is recorded.

- **Schema & Content:**  
  - The exact schema of the database is provided below.  
  - A complete list of questions and their corresponding answer options is also provided.

---

**2. Role and Responsibilities**  
- **Primary Task:**  
  - Evaluate the user's request and determine whether it is possible to answer with the available data.

- **Important Constraint:**  
  - The database stores the aggregated total responses per school only (not per individual student).  
  - **Implication:** It is impossible to answer queries that attempt to link responses across multiple questions (for example, “How many students who answered question X also answered question Y?”).

- **Decision Process:**  
  - **If the request is unanswerable:**  
    - Respond to the user with a message that begins with the words **"FINAL ANSWER"**, followed by an explanation stating that the query cannot be fulfilled due to the limitations of the database.
  - **If the request is answerable:**  
    - Prepare a proposal for the human assistance tool with a suggested set of tables, columns, and fields that the SQL agent can use to formulate the query.
    - Once the expert provides feedback, forward a well-formulated prompt—including the user’s request and the designated fields, columns, and tables—to the SQL agent.

---

**3. Examples of Query Types**  
- **Answerable Queries:**  
  - Requests that ask for aggregated totals per question or answer option by school and/or year.
  - Example: “What is the total number of male and female responses for question X in school Y for the year 2015?”

- **Unanswerable Queries:**  
  - Requests that require linking responses between two or more different questions.
  - Example: “How many students who answered question A also answered question B?”

---

**4. Instructions for Query Processing**  
- **For Unanswerable Requests:**  
  - Begin the response with **"FINAL ANSWER"** and include an explanation indicating that the requested linkage between multiple questions cannot be provided given the aggregated nature of the data.

- **For Answerable Requests:**  
  - Draft a proposal outlining:
    - The specific tables, columns, and fields available for querying.
    - A clear and well-formulated prompt that includes the user’s request and the suggested data structure.
  - Submit this proposal to the human assistance tool.
  - Incorporate expert feedback and then pass the finalized SQL query instructions to the SQL agent.


--------------------------------------------------------------------------------  
Below are the exact database schema and valid response strings. Use these precise values in your queries to ensure accurate matching with the database structure.

DB Schema and Values Format :

Table: schools
--------------
Column: school_id | Type: INTEGER
Column: school_name | Type: VARCHAR(100)

Table: questionnaires
---------------------
Column: questionnaire_id | Type: VARCHAR(100)
Column: school_id | Type: INTEGER
Column: type_id | Type: INTEGER
Column: year | Type: INTEGER
Column: total_graduates | Type: INTEGER
Column: questionnaire_completed | Type: INTEGER
Column: percent_answered | Type: DOUBLE PRECISION
Column: female_count | Type: INTEGER
Column: male_count | Type: INTEGER
ForeignKey: ReadOnlyColumnCollection(questionnaires.type_id) references types.type_id
ForeignKey: ReadOnlyColumnCollection(questionnaires.school_id) references schools.school_id

Table: types
------------
Column: type_id | Type: INTEGER
Column: type | Type: VARCHAR(100)

Table: questions
----------------
Column: question_id | Type: INTEGER
Column: question_text | Type: TEXT
Column: type_id | Type: INTEGER
ForeignKey: ReadOnlyColumnCollection(questions.type_id) references types.type_id

Table: responses
----------------
Column: response_id | Type: INTEGER
Column: questionnaire_id | Type: VARCHAR(50)
Column: question_id | Type: INTEGER
Column: gender | Type: VARCHAR(1)
Column: response_option | Type: VARCHAR(100)
Column: summed_total_all_students_responses | Type: INTEGER
ForeignKey: ReadOnlyColumnCollection(responses.question_id) references questions.question_id
ForeignKey: ReadOnlyColumnCollection(responses.questionnaire_id) references questionnaires.questionnaire_id

Values Exact match Format :

Years : 
From 2004 to 2019

school_id : School Names 
1: Aux quatre vents
2: Centre La Fontaine
3: Secondaire Népisiguit
4: Louis-Mailloux
5: Marie-Esther
6: Roland-Pépin
7: W.-A.-Losier

type_id : type
1: Questions Générales
2: SD-Renseignements Socio-Démographiques
3: ED- Éducation PostSecondaire
4: MT- Marché du travail
5: RE-Attente d'emploi / Recherche d'emploi / Sans emploi

Questions and responses for exact match :

Question: À la fin de vos études postsecondaires, si vous aviez une offre d'emploi dans la Péninsule acadienne, est-ce que vous l'accepteriez ?
: ['Non', 'Non répondu', 'Je ne sais pas', 'Oui']

Question: Au cours des 2 dernières années, combien d'employeurs avez-vous eus ?
: ['Un', 'Aucun', 'Plus de deux', 'Non répondu', 'Deux']

Question: Au cours des deux dernières années, combien de mois d'expérience de travail avez-vous accumulés ?
: ['Moins de 6 mois', '12 à 18 mois', '18 à 24 mois', '6 à 12 mois', 'Non répondu']

Question: Avez-vous complété vos études ?
: ['Non répondu', 'Non', 'Oui']

Question: Avez-vous eu un emploi au cours des 2 dernières années ?
: ['Non répondu', 'Non', 'Oui']

Question: Avez-vous l'intention de vous établir dans la Péninsule acadienne au cours des cinq prochaines années ?
: ['Non répondu', 'Non', 'Ne sais pas', 'Oui']

Question: Dans quel domaine d'études étiez-vous inscrit/e ?
: ['Mathématiques, informatique et sciences physiques', 'Professions de la santé et technologies connexes', 'Commerce, gestion et administration des affaires', 'Beaux-arts et arts appliqués', 'Lettres, sciences humaines et disciplines connexes', 'Enseignement, loisirs et orientation', 'Sciences sociales et disciplines connexes', 'Non repondu', 'Génie et sciences appliquées', 'Sans spécialisation', 'Non répondu', 'Techniques et métiers des sciences appliquées']

Question: Dans quel type d'institution avez-vous étudié ?
: ['Universitaire', 'Non répondu', 'Collégial', 'École privée']

Question: Dans quelle catégorie de profession ou métier travaillez-vous ?
: ['Commerce de détail', 'Arts, spectacles et loisirs', 'Fabrication', 'Services professionnels, scientifiques et technique', 'Agriculture, forestierie, pêche et chasse', "Services admistratifs, services de soutien, services de gestion de déchets et d'assainissement", 'Finance et assurances ', 'Commerce en gros', "Insdustrie de l'information - industrie culturelle", 'Transport et entreposage', 'Non répondu', 'Services publiques', 'Hébergement et services de restauration', "Gestion de sociétés et d'entreprises", 'Autre services (sauf administration publique)', "Services d'enseignement", 'Soins de santé et assistance sociale', 'Extraction minière, extraction de pétrole et de gaz ', 'Services immobiliers, service de location', 'Construction', 'Administration publiques']

Question: Dans quelle province canadienne ou ailleurs avez-vous étudié ?
: ['(Angleterre)', '(Québec)', '(Manitoba)', '(Non répondu)', '(Alberta)', '(Saskatchewan)', '(Colombie-Britannique)', '(Ontario)', '(France)', '(Île-du-Prince-Édouard)', '(Nouveau-Brunswick)', '(Nouvelle-Écosse)', '(Terre-Neuve)']

Question: Dans quelle province canadienne ou autre pays aimeriez-vous travailler ?
: ['Alberta', 'Saskatchewan', 'Autre', 'Colombie-Britanique', 'Ontario', 'Terre-Neuve', 'Nouveau-Brunswick', 'Québec', 'Non répondu', 'non répondu']

Question: Dans quelle province canadienne ou autre pays travaillez-vous ?
: ['(Angleterre)', '(Québec)', '(Manitoba)', '(Non répondu)', '(Territoires-du-Nord-Ouest)', '(Alberta)', '(Saskatchewan)', '(Colombie-Britannique)', '(Nunavut)', '(Ontario)', '(France)', '(Île-du-Prince-Édouard)', '(Nouveau-Brunswick)', '(Suisse)', '(Nouvelle-Écosse)', '(Terre-Neuve)']

Question: Dans quelle région de la province aimeriez-vous travailler ?
: ['Nord-Est', 'Saskatchewan', 'Nord', 'Sud-ESt', 'Nord-Ouest', 'Sud', 'Colombie-Britannique', 'Autre', 'Terre-Neuve', 'Nouveau-Brunswick', 'Sud-Est', 'Non répondu', 'Sud-Ouest']

Question: Dans quelle région de la province travaillez-vous ?
: ['Nord-Est', 'Ouest', 'Nord', 'Nord-Ouest', 'Sud', 'Autre', 'Centre', 'Est', 'Non Répondu', 'Sud-Est', 'Non répondu', 'Sud-Ouest', 'non répondu']

Question: De quelle localité de la Péninsule acadienne êtes-vous originaire ?
: ['AIRDRIE', 'ROBICHAUD SETTLEMENT', 'POINTE-A-BOULEAU', 'ANSE-BLEUE', 'Granby', 'BALMORAL', 'GATINEAU', 'Lugar', 'HAUT-SHIPPAGAN', 'PARIS, FRANCE', 'PAQUETVILLE', 'PETIT-TRACADIE', 'ST SAUVEUR', 'DALHOUSIE', 'Saint-Sauveur', 'HAUT-LAMÈQUE', 'SAINTE-MARIE-SAINT-RAPHAEL', 'BERESFORD', 'MISCOU', 'SORMANY', 'TREMBLAY', 'ATHOLVILLE', 'ST- ARTHUR', 'LAVILLETTE', 'PETITE RIVIÈRE DE L ÎLE', 'MIRAMICHI ROAD', 'POKEMOUCHE', 'VILLAGE ST-LAURENT', 'XX', 'ST-AMATEUR', 'STE-ROSE', 'ILES DE LA MADELAINE', 'TRACADIE-SHEILA', 'BAIE DE PETIT-POKEMOUCHE', 'CAISSIE ROAD', 'CANTON DES BASQUE', 'SOUTH TETAGOUCHE', 'BEAVERDAM', 'BURNSVILLE', 'DUGUAYVILLE', 'GLOUCESTER', 'Buckland', 'POINTE À TOM', 'FAIR-ISLE', 'PETIT- ROCHER', 'NEW CARLISLE', 'TRACADIE-BEACH', 'ST-LÉOLIN', 'BEAUPORT', 'PETIT LAMÈQUE', 'DUGAS', 'ROBERTVILLE', 'SAINT-ARTHUR', 'HAUT-ST-SIMON', 'LOSIER SETTLEMENT', 'ROCHEVILLE', 'MCLEOD', 'PEACE RIVER', 'BURN CHURCH', 'POINTE DES ROBICHAUD', 'SAUMAREZ', 'DALHOUSIE JUNCTION', 'GLENCOE', 'Mckendrick', 'Ste-Thérèse', 'CHIASSON OFFICE', 'MAISONNETTE', 'CHURCH POINT', 'VAL-DOUCET', 'Saint Jéröme', 'QUÉBEC', 'LAMÈQUE', 'BOIS-BLANC', 'OTTAWA', 'Saint Quentin', 'SHIPPAGAN', 'SAINTE-ANNE', 'NORTH TETAGOUCHE', 'TILLEY-ROAD', 'MASCOUCHE', 'MONTRÉAL', 'DAUVERSIERE', 'ST-SIMON', 'ST-WILFRED', 'STE-LOUISE', 'SAINTE-ROSE', 'BAS-PAQUETVILLE', 'VAL-COMEAU', 'ALDERWOOD', 'INKERMAN', 'LEECH', 'BURNT CHURCH', 'Saint-Amateur', 'Nouvelle (Québec)', 'Ontario', 'BAS-CARAQUET', 'SAINTE-LOUISE', 'RANG ST-GEORGES', 'SAINT-LÉOLIN', 'RIO GRANDE', 'TRUDEL', 'PETIT-ROCHER', 'PETIT PAQUETVILLE', 'Saint Jean', 'PETIT POKEMOUCHE', 'Saint-Maure', 'STYMIEST ROAD', 'DUNLOP', 'ST-JEAN', 'BENOIT', 'STE-ANNE GLOUCESTER', 'Matapédia', 'ST-MAURE', 'YELLOW KNIFE', 'BOIS-GAGNON', 'NÉGUAC', 'DUNDEE', 'BIG RIVER', 'MADRAN', 'HAUT-PAQUETVILLE', 'SAINT-WILFRED', 'BLACK-ROCK', 'DUGAS-OFFICE', 'NOUVELLE-ÉCOSSE', 'TIDE HEAD', 'SAINT-SIMON', 'CARAQUET', 'NON RÉPONDU', 'DIEPPE', 'POINTE LA NIM', 'VAL DAMOUR', 'Kingston', 'LAGACÉVILLE', 'ALLAINVILLE', 'SIX-ROADS', 'Cantley', 'EEL RIVER COVE', 'Ancienne Lorette', 'POINTE-SAUVAGE', 'Beauharnois', 'CHEMIN DU COTEAU ', 'ALCIDA', 'Saint-Laurent', 'COTEAU-ROAD', 'POINTE CANOT', "VAL D'OR", 'JOLIETTE', 'NOTRE-DAME-DES-ÉRABLES', 'POINTE-BRÛLÉE', 'INKERMAN-FERRY', 'NIGADOO', 'FOUR ROADS', 'RIVIÈRE-À-LA-TRUITE', 'NEW JERSEY', 'PETIT ROCHER', 'HAUT-SHEILA', 'BAS-NÉGUAC', 'Ste Irenée', 'ST-QUENTIN', 'PONT-LANDRY', 'HACHEYVILLE', 'CHICOUTIMI', 'Malauze', 'BRANTVILLE', 'PETIT-ROCHER NORD', 'ALLARDVILLE', 'POKESUDIE', 'EVANGÉLINE', 'MALTEMPEC', 'ST-IRÉNÉE', 'ALLADVILLE', 'GRANDE-ANSE', 'PETIT-ROCHER OUEST', 'LAVAL', 'HULL', 'POINTE-VERTE', 'MIRAMICHI', 'ST-LEOLIN', 'QUISPAMSIS', 'COMEAU SETTLEMENT', 'MIDDLE RIVER', 'Sainte-Thérèse Sud', 'CAMPBELLTON', 'SUDBURY', 'VILLAGE-DES-POIRIERS', 'Point La Nim', 'OAK POINT', 'BLAIR ATHOL', 'ROGERSVILLE', 'EDMUNSTON', 'TETAGOUCHE NORD', 'GAUVREAU', 'LE GOULET', 'MONCTON', 'Sainte-Rosette', 'PETIT-SHIPPAGAN', 'SAVOIE LANDING', 'BARTIBOG', 'BERTRAND', 'ROUGH WATERS', 'GRAND SAULT', 'ST-ISIDORE', 'LAPLANTE', 'NICHOLAS DENYS', 'LANDRY OFFICE', 'EEL RIVER CROSSING', 'ST-HYACINTHE', 'BOUCTOUCHE', 'VILLAGE BLANCHARD', 'PIGEON-HILL', 'CAP-BATEAU', 'Prévost', 'TETAGOUCHE SUD', 'RIVIÈRE-DU-PORTAGE', 'EEL RIVER', 'PONT-LA-FRANCE', 'ST-LÉONARD', 'Richardsville', 'TABUSINTAC', 'PETIT-ROCHER,', 'POINTE-ALEXANDRE', 'CHARLO', 'STE-ROSETTE', 'BATHURST', 'POINTE A LA CROIX', 'PETIT-ROCHER SUD', 'HAUT RIVIÈRE-DU-PORTAGE', 'Courtney', 'ST-PONS', 'St-Hilaire', 'ST-SAUVEUR', 'SAINTE-CÉCILE', 'BELLEDUNE']

Question: Depuis combien d'années demeurez-vous à l'extérieur de la Péninsule acadienne ?
: ["moins d'un an", 'Deux ans', 'Un an', 'Trois ans', 'Plus de quatre', 'Non répondu', 'Quatre ans']

Question: Depuis combien de temps êtes-vous sans emploi ?
: ['Moins de 6 mois', 'Plus de 2 ans', '1 à 2 ans', '6 à 12 mois', 'Non répondu']

Question: Depuis septembre dernier, avez-vous occupé un emploi rénuméré ?
: ['5. Autres', '1. Non', "4. à l'occasion (ex. 1-2 fois par semaine)", '2. Temps partiel fin de semaine', '3. Temps partiel à la semaine']

Question: Durant vos études postsecondaires, quelles seront vos sources de financement ?
: ['3. Contribution des parents', '2. Bourses étudiantes', '4. Économie personnelle', '1. Prêt étudiant', '5. Travail à temps partiel', '6. Autre']

Question: Durant vos études quelles étaient vos sources de financement ?
: ['Bourse étudiants', 'Prêts étudiant', 'Économie personnelle', 'Travail à temps partiel', "Plan enregistré d'épargne-études", 'Autre', 'Travail à temps plein', 'Prêts/bourse étudiant', 'Bourse étudiante', "Programme d'aide fédéral", 'Travaillle à temps plein', "Plan enregistré d'épargnes-études", "Programme d'aide provincial", 'Travaille à temps partiel', 'Non répondu', 'Contribution des parents']

Question: En général, êtes-vous satisfait/e de vos résultats scolaires ?
: ['Satisfait/e', 'Très insatisfait/e', 'Insatisfait/e', 'Non répondu', 'Très satisfait/e']

Question: En général, quels sont vos résultats scolaires en 10e 11e et 12e année ?
: ['de 81 % à 94 %', 'de 56 % à 65 %', 'de 95 % à 100 %', 'moins de 55 %', 'de 66 % à 80 %', 'Non répondu']

Question: En général, vos cours sont de quel niveau scolaire ?
: ['Adaptation scolaire', 'PCE', 'Régulier', 'Compétences essentielles', 'Choix', 'Non répondu', 'Modifié']

Question: Envisagez-vous de changer d'emploi dans la prochaine année ?
: ['Non répondu', 'Non', 'Oui']

Question: Est-ce que le fait de terminer votre 12e année est important pour vous ?
: ['Important', 'Peu important', 'Pas du tout important', 'Très important', 'Non répondu']

Question: Est-ce que les services d'orientation que vous avez utilisés ont répondu à vos besoins ?
: ['Pas utilisés', 'Peu', 'Bien', 'Pas du tout', 'Très bien', 'Non répondu']

Question: Êtes-vous disponible pour travailler maintenant ?
: ['Non', 'Non répondu', 'Oui']

Question: Existe-t-il des obstacles qui vous empêchent de poursuivre des études ?
: ['Non répondu', 'Non', 'Oui']

Question: Indiquez dans quel domaine vous avez l'intention de poursuivre vos études postsecondaires ?
: ['Mathématiques, informatique et sciences physiques', 'Sciences agricoles et biologiques et services de la nutrition et de l&rsquo;alimentation', 'Professions de la santé et technologies connexes', 'Commerce, gestion et administration des affaires', 'Beaux-arts et arts appliqués', 'Lettres, sciences humaines et disciplines connexes', 'Enseignement, loisirs et orientation', 'Sciences sociales et disciplines connexes', 'Génie et sciences appliquées', 'Sans spécialisation', 'Non répondu', 'Techniques et métiers des sciences appliquées']

Question: Indiquez le nom de l'institution que vous pensez fréquenter l'an prochain ?
: ['Université de Sherbrooke', 'University Mount Allison', 'Notre-Dame-de-Foy', 'Université Western', 'Recording Arts Canada', 'Oulton College', 'CITÉ COLLÉGIALE', 'Eastern College', 'MEDES Institutes', 'SALON CAROUSSEL', 'Université de Moncton - campus Moncton', 'NDI', 'Université Laval', 'Académie de Police (Régina)', 'CCNB (PA)', 'UNIVERSITE LAVAL', 'NBCC Miramichi Campus', 'CCNB MIRAMICHI', 'Nav Canada', 'College de la Garde Cotière Canadienne', 'Université St-Francis Xavier', 'ESSENCE INSTITUT', 'UNIVERSITÉ DE MONTRÉAL', 'École de coiffure Rose-Hélène', 'Autre', 'CCNB Moncton', 'OULTON COLLEGE', '2. Bourses étudiantes', '4. Économie personnelle', 'ACADEMY OF LEARNING MIRAMICHI', 'Brenda dog grooming', 'Académie paramédicale de lAtlantique', 'Universdad de Boenos', 'NAIT', '1. Prêt étudiant', "UNIVERSITÉ D'OTTAWA", 'UNIVERSITÉ STE-ANNE', 'CCNB (Bathurst)', 'ÉCOLE DES PÊCHES', 'THE ART INSTITUTE OF TORONTO', '6. Autre', 'MEMORIAL UNIVERSITY', 'St-Thomas University', 'Cégep', 'Université d Acadia', 'COLLÈGE LASALLE', 'École nationale opération de machinerie lourde', 'Université de Guelph', 'Mount Allison', 'Académie La Coupe Plus', 'CCNB (Campbelton)', 'Université de Moncton - campus Edmunston', 'Université McGill', 'COLLÈGE BORÉAL', 'College de technologie forestiere des Maritimes', 'COLLÈGE MILITAIRE ROYAL DU CANADA', 'CCNB Campbellton-Cours distance', 'Centre de formation de la Haute-Gaspésie', 'Institut humanitaire', 'University of Mobile', 'Le Collège Atlantic de Massage thérapeutique', 'centre de formation professionnelle , Trois Rivières. qc', 'CRIF', 'Université de Montréal', 'CCNB (Edmunston)', 'CCNB (Autre)', 'CITE COLLÉGIALE', 'Dalhousie  University', "Université St Mary's", "ECOLE DES MÉTIERS DE L'INDUSTRIE DE LA CONSTRUCTION", "INST. TECH.D'AGROALIMENTAIRE", "Université de l'Ile du Prince Edouard", '3. Contribution des parents', 'Campus Notre Dame de Foy', 'NBCC MONCTON', 'UCAM', 'Université de Moncton - campus Bathurst', 'Université du Nouveau-Brunswick', 'Université de Concardia', 'CCNB Fredericton', 'Collège Acadie, Université Ste-Anne', 'Amor Esthétique', 'Université Hearst', 'Maritime College of forest technololgy', 'ÉCOLE DE COIFFURE DE LA PÉNINSULE', 'UNIVERSITÉ LAURENTIENNE', 'UNIVERSITÉ DE MONCTON(CAMPUS DE SHIPPAGAN ET MONCTON)', 'École National du Théatre du Canada', 'Da Vinci College', 'Cégep de Ste Foy', 'Université de Moncton - campus Shippagan', 'Université de Toronto, campus St-George', 'Institut Maritime du Québec', 'JON RAYMOND', 'MEDES', 'Université du Québec', 'Collège Atlantique de massages thérapeutique', 'CEGEP DE LAPOCATIÈRE', 'Amoura Esthetic', '5. Travail à temps partiel', 'Non répondu', 'CCNB (Dieppe)', 'St Thomas University', 'Université du Canadienne', "BRENDA'S ACADEMY OF PROFESSIONNAL DOG GROOMING", 'Holland College', 'Université de Dalhousie', 'NBCC', 'Université libre de Bruxelles', 'Medavie HealtEd', 'Complexe Capital Hélicoptère', 'NBCC Monction', 'Bathurst Hair academy', 'Collège Merici', 'UNB', 'Cégep de Rimouski']

Question: Indiquez les deux cours que vous avez les plus appréciés durant votre secondaire
: ['46. Sciences familiales', "45. Sc. De l'environnement", '52. Technologie', '18. Droit', '22. Éducation coopérative', '36. IPEJ', '49. Sciences sociales', '10. Biologie', '29. Français', '39. Mécanique des moteurs', "35. Intro. Science de l'info.", '48. Sciences humaines', '50. Systèmes mécaniques', '14. Chimie', '42. Physique', '47. Sciences générales', '37. Leadership', "54. Trait. De l'information", '3. Alimentation et nutrition', '44. Saisie sur clavier', '12. Braille Nameth', '20. Éduc. Du consommateur', '5. Anglais', '53. Textiles et habillement', '9. Atelier de métaux', '2. Agriculture', '40. Menuiserie', '19. Économie', '6. Art dramatique', '33. Informatique appliqué', '1. Act.Quot.(handicap vis.)', '43. Relations familiales', '11. Braille abrégé', '13. Charpenterie', '30. Géographie', '28. Formation personnelle', '17. Développement humain', '15. Comptabilité', '27. Espagnol', '51. Techno. (handicap visuel)', '26. Entreprenariat', '7. Arts Industriels', '34. Initiation au travail', '41. Musique', '25. Électronique', '24. Électricité', '31. Histoire', '32. Informatique', '38. Mathématiques', '8. Arts visuels', '16. Dessin industriel', '4. Allemand', '21. Éducation aux valeur', '23. Éducation physique']

Question: Lorsque vous aurez terminé vos études postsecondaires, indiquez où vous prévoyez travailler ?
: ['4. Charlotte (N.-B.)', '19. Nouvelle-Écosse', '20. Ontario', '26. Yukon', '35. Canada', '17. IPE', '31. Autre', '11. Restigouche (N.-B.)', '24. Alberta', '27. Territoire du Nord-Ouest', '16. York (N.-B.)', '33. Autre(s) province(s)', '23. Saskatchewan', '6. Kent (N.-B.)', '1. Dans la Péninsule acadienne', '32. Nouveau-Brunswick', '12. Sunbury (N.-B.)', '2. Albert (N.-B.)', '10. Queens (N.-B.)', '21. Québec', '18. Terre-Neuve', '3. Carleton (N.-B.)', '9. Northumberland (N.-B.)', '7. Kings (N.-B.)', '30. Angleterre', '29. France', '8. Madawaska (N.-B.)', '14. Victoria (N.-B.)', '28. État-Unis', '13. Saint-Jean (N.-B.)', '22. Manitoba', '5. Gloucester (N.-B.)', '15 Westorland (N.-B.)', '34. Autre(s) pays', '25. Colombie-Britanique']

Question: Où demeurez-vous présentement ?
: ['Péninsule acadienne', 'Ailleurs au N.-B.', 'Ailleurs au Canada', 'Dans un autre pays', 'Non répondu']

Question: Pour quelle raison avez-vous quitté la Péninsule acadienne ?
: ['Travail', 'Non répondu', 'Étude', 'Autre']

Question: Présentement êtes-vous en attente ou à la recherche d'emploi ?
: ["Recherche d'emploi", 'Non', "Attente d'emploi", 'Non répondu', 'Oui']

Question: Que consirérez-vous comme étant votre principale occupation en ce moment ?
: ["Recherche d'emploi / sans emploi", 'Termine mes études secondaires', 'Études universitaires', 'Formation professionnelle ', 'Retours aux études secondaires', "Recherche d'emploi", "Recherche d'emploi / Sans emploi", 'Autre', 'Marché du travail', 'Armée', 'Programme jeunesse', 'Non répondu', 'Termine mes études postsecondaires', 'non répondu']

Question: Que prévoyez-vous faire en septembre prochain ?
: ['Marché du travail', 'Année sabbatique ', 'Programmes jeunesse', 'Ligne de hockey', 'Formation professionnelle ', 'Retours aux études secondaires', 'Voyage pour Missionnaire ', 'Cégep ', 'Études collégiales ', 'Ne sais pas ', 'Non répondu', 'Voyager ', 'Études universitaires', 'Écoles des pêches ', 'Indécis du domaine souhaiter ', 'Programme jeunesse', 'Cours se donne seulement aux 2 ans ', 'Pas assez de bonnes notes', 'Cours par correspondance ', 'Armée']

Question: Quel est votre état civil présentement ?
: ['Divorcé/e', 'Veuf / veuve', 'Célibataire', 'Marié/e', 'Conjoint/e de fait', 'Séparé/e', 'Non répondu']

Question: Quel est votre niveau d'éducation le plus élevé ?
: ['Formation personnelle', 'Secondaire (12 ième année)', 'Secondaire', 'Secondaire non-complété', 'DEG non-complété', 'Universitaire (baccalauréat)', 'non répondu', 'Formation professionnelle', 'Universitaire (Doctorat)', 'Non répondu', 'Collégial', 'Universitaire (Maîtrise)', 'DESPA complété', 'DESPA non complété', 'Universitaire (Baccalauréat)', 'DESPA Complété', 'DEG complété', 'Secondaire (12 ième complété)', 'Autre']

Question: Quel était le niveau de votre programme d'études ?
: ['Troisième cycle ( doctorat )', 'Deuxième cycle ( maîtrise )', 'Premier cycle universitaire( bacc., certificat)', 'Diplôme', 'Certificat', 'Non repondu', 'Cours préparatoire ( 13e année )', 'Sciences de la santé', 'Non répondu']

Question: Quel était votre régime d'études ?
: ['Temps plein', 'Non répondu', 'Temps partiel']

Question: Quel moyen de transport local pensez-vous utiliser le plus régulièrement pendant votre première année d'études ?
: ['11. Covoiturage', '10. Taxi', '7. Voiture frère/sœur', '3. Voiture de mes parents', '1. À pied', '8. Transport en commun', '2. Ma propre voiture/motocyclette', '6. Métro', '5. Autobus', '12. Autre', '9. Bicyclette', '6. Autre', "4. Voiture d'un ami/e ami/e"]

Question: Quel type d'emploi recherchez-vous ?
: ['Commerce de détail', 'Arts, spectacles et loisirs', 'Fabrication', 'Services professionnels, scientifiques et technique', 'Agriculture, forestierie, pêche et chasse', "Services admistratifs, services de soutien, services de gestion de déchets et d'assainissement", 'Finance et assurances ', 'Commerce en gros', "Insdustrie de l'information - industrie culturelle", 'Transport et entreposage', 'Non répondu', 'Services publiques', 'Hébergement et services de restauration', "Gestion de sociétés et d'entreprises", 'Autre services (sauf administration publique)', "Services d'enseignement", 'Soins de santé et assistance sociale', 'Extraction minière, extraction de pétrole et de gaz ', 'Construction', 'Administration publiques']

Question: Quelle est la principale raison pour laquelle vous envisagez de changer d'emploi ?
: ['Je veux travailler plus près de chez moi', "J'ai besoin d'un changement", 'Mon travail actuel ne correspond pas à mon expérience et à ma formation', 'Je travaille à temps plein et je préférerais travailler à temps partiel', "Je recherche la sécurité d'emploi", 'Je désire retourner aux études']

Question: Quelle est la principale raison pour laquelle vous n'avez pas travaillé durant cette période ?
: ["Attente d'un rappel au travail après mise à pied", 'Invalidité', 'Accidenté', "Rémunération insuffisante pour quitter l'A-S ou A-E", 'Changement de propriétaire', 'Chômage / travail saisonnier', 'Déménagement', 'Pas de voiture', 'Conflit de travail', 'Incapacité de trouver un emploi lié à ma formation', 'Accouchement', "Attente du début d'un emploi", 'Chômage de quelques semaines', 'Retour aux études', 'Responsabilité personnelles ou familiales', 'Non répondu', 'Maladie']

Question: Quelle est la raison pour laquelle vous ne prévoyez pas poursuivre des études postsecondaire ?
: ['Accès au marché du travail', 'Peur de ne pas réussir', 'Cours non disponible', 'Aucune idée du domaine à choisir', 'Manque d’intérêt pour les études/formation', 'Je travaille ou prévois travailler', 'Autre', 'Prendre un an', 'Armée', 'Problèmes monétaires', 'Non répondu']

Question: Seriez-vous intéressé à recevoir des l'information concernant une expérience de travail dans votre communauté ?
: ['Non', 'Non répondu', 'Oui']

Question: Seriez-vous intéressé/e à faire un retour aux études ?
: ['Non', 'Ne sais pas', 'Présentement aux études', 'Non répondu', 'Oui']

Question: Seriez-vous intéressé/e à recevoir de l'information pour vous aider dans la poursuite de vos études ?
: ['Non répondu', 'Non', 'Oui']

Question: Seriez-vous intéressé/e par une expérience de travail dans votre communauté ?
: ['Non répondu', 'Non', 'Oui']

Question: Sur le plan de vos relations avec le personnel de l'école, êtes-vous satisfait/e de votre séjour au secondaire ?
: ['Satisfait/e', 'Très insatisfait/e', 'Insatisfait/e', 'Non répondu', 'Très satisfait/e']

Question: Sur le plan de vos relations avec les autres élèves, êtes-vous satisfait/e de votre séjour au secondaire ?
: ['Satisfait/e', 'Très insatisfait/e', 'Insatisfait/e', 'Non répondu', 'Très satisfait/e']

User Request: {user_request}"""

# Function to analyze the user request
def analyze_request(state: DatabaseQueryState):
    """Analyze whether the user request can be answered with the database"""
    
    # Format system message
    user_request = state['user_request']
    system_message = query_analysis_instructions.format(user_request=user_request)
    
    # Create structured LLM call
    structured_llm = llm.with_structured_output(AnalysisResult)
    
    # Generate analysis
    analysis_result = structured_llm.invoke([SystemMessage(content=system_message)])
    
    # Return updated state
    return {"analysis_result": analysis_result}

# Function for human feedback step
def human_feedback(state: DatabaseQueryState):
    """No-op node that should be interrupted for human feedback"""
    pass

# Function to finalize query instructions
def finalize_query(state: DatabaseQueryState):
    """Create final query instructions based on the analysis and human feedback"""
    
    user_request = state['user_request']
    analysis_result = state['analysis_result']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    
    # System message for finalizing query
    system_message = f"""Based on the user's request, the analysis of whether it's answerable, and human expert feedback, create the final SQL query instructions.

User Request: {user_request}

Analysis: {analysis_result.explanation}

Proposed Tables: {', '.join(analysis_result.query_proposal.tables) if analysis_result.query_proposal else 'None'}
Proposed Columns: {', '.join(analysis_result.query_proposal.columns) if analysis_result.query_proposal else 'None'}
Proposed Fields: {', '.join(analysis_result.query_proposal.fields) if analysis_result.query_proposal else 'None'}

Human Expert Feedback: {human_analyst_feedback}

Create a clear and well-formatted set of instructions for the SQL agent."""

    # Generate final instructions
    response = llm.invoke([SystemMessage(content=system_message)])
    
    # Return updated state
    return {"final_query_instructions": response.content}

# Function to handle unanswerable queries
def handle_unanswerable(state: DatabaseQueryState):
    """Format response for unanswerable queries"""
    
    analysis_result = state['analysis_result']
    
    # Create "FINAL ANSWER" response
    final_answer = f"FINAL ANSWER: {analysis_result.explanation}"
    
    # Return updated state
    return {"final_query_instructions": final_answer}

# Conditional edge function to route based on answerability
def route_after_analysis(state: DatabaseQueryState):
    """Determine next step based on whether query is answerable"""
    
    analysis_result = state['analysis_result']
    
    if analysis_result.is_answerable:
        return "human_feedback"
    else:
        return "handle_unanswerable"

# Conditional edge function after human feedback
def should_continue(state: DatabaseQueryState):
    """Route based on presence of human feedback"""
    
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    
    if human_analyst_feedback:
        return "finalize_query"
    
    # If no human feedback, end
    return END

# Build the graph
builder = StateGraph(DatabaseQueryState)

# Add nodes
builder.add_node("analyze_request", analyze_request)
builder.add_node("human_feedback", human_feedback)
builder.add_node("finalize_query", finalize_query)
builder.add_node("handle_unanswerable", handle_unanswerable)

# Add edges
builder.add_edge(START, "analyze_request")
builder.add_conditional_edges("analyze_request", route_after_analysis, 
                             ["human_feedback", "handle_unanswerable"])
builder.add_conditional_edges("human_feedback", should_continue,
                             ["finalize_query", END])
builder.add_edge("finalize_query", END)
builder.add_edge("handle_unanswerable", END)

# Compile the graph
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# Example usage
# config = {"configurable": {"thread_id": "123"}}
# result = graph.invoke({"user_request": "What is the total number of male students who answered 'Yes' to Question 3 in 2018?"}, config)

# # Display the graph
# import os
# import sys
# import subprocess

# # Get the PNG data from your graph
# png_data = graph.get_graph(xray=1).draw_mermaid_png()

# # Write the PNG data to a file
# filename = "graph.png"
# with open(filename, "wb") as f:
#     f.write(png_data)

# # Open the file using the default image viewer based on your OS
# if sys.platform.startswith("win"):
#     os.startfile(filename)
# elif sys.platform == "darwin":  # macOS
#     subprocess.call(["open", filename])
# else:  # Linux and others
#     subprocess.call(["xdg-open", filename])

# # Input
# #user_request = "Nombre d'etudiant qui ont de bons resultats scolaires et vont sur pieds a l'ecole" 
# user_request = "Nombre d'etudiant qui ont de bons resultats scolaires en 2018"
# #user_request = "Bonjour"

# thread = {"configurable": {"thread_id": "1"}}

# # Run the graph until the first interruption
# for event in graph.stream({"user_request":user_request}, thread, stream_mode="values"):
#     print(event)

# state = graph.get_state(thread)
# print(state.next)

# # If we are satisfied, then we simply supply no feedback
# further_feedack = "seems correct"
# graph.update_state(thread, {"human_analyst_feedback": 
#                             further_feedack}, as_node="human_feedback")

# for event in graph.stream(None, thread, stream_mode="updates"):
#     print("--Node--")
#     node_name = next(iter(event.keys()))
#     print(node_name)

# final_state = graph.get_state(thread)
# print(final_state.values.get('final_query_instructions'))
def invoke_our_graph(user_input, callables, thread_id):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return graph.invoke(
        {"user_request": user_input},
        config={"callbacks": callables, "configurable": {"thread_id": thread_id}},
    ), graph