# AI Assistant with RAG

Un'applicazione Streamlit che integra tecnologie di Retrieval-Augmented Generation (RAG) per creare due strumenti: un Document Q&A e un Meal Planner intelligente.

## 📖 Introduzione

Questo progetto è un'applicazione Streamlit che dimostra l'utilizzo di tecniche di Retrieval-Augmented Generation (RAG) attraverso due moduli principali:

- Document Q&A: Un sistema che permette agli utenti di caricare documenti (PDF, DOCX, TXT) e porre domande sul loro contenuto, ricevendo risposte precise basate sui contenuti del documento stesso.
- Meal Planner: Un assistente che consente agli utenti di caricare ricette in formato JSON, fare domande sulle ricette, generare piani alimentari personalizzati e creare liste della spesa ottimizzate.

## ✨ Caratteristiche

- Interfaccia intuitiva con Streamlit
- Elaborazione avanzata dei documenti con chunking intelligente e memorizzazione vettoriale
- Supporto per documenti multipli (PDF, DOCX, TXT)
- Domande e risposte basate sul contesto tramite RAG
- Generazione di riassunti dei documenti
- Pianificazione pasti personalizzata basata su preferenze dietetiche
- Generazione automatica di liste della spesa
- Supporto multi-piattaforma

## 🔍 Cos'è RAG?

Retrieval-Augmented Generation (RAG) è un paradigma all'avanguardia nell'intelligenza artificiale che combina:

- Retrieval (Recupero): La capacità di cercare e recuperare informazioni pertinenti da una base di conoscenza o documenti specifici.
- Augmentation (Arricchimento): L'arricchimento dei modelli linguistici con conoscenze specifiche recuperate.
- Generation (Generazione): La generazione di risposte coerenti e contestuali basate sia sulla conoscenza generale del modello che sulle informazioni recuperate.

RAG risolve alcuni dei problemi più significativi dei modelli linguistici tradizionali:

- Conoscenza limitata o obsoleta del modello base
- Mancanza di citazioni o riferimenti per le informazioni fornite
- Allucinazioni (generazione di informazioni inesatte)
- Costi elevati per il fine-tuning completo dei modelli

## Nel nostro sistema, RAG funziona così:

I documenti vengono suddivisi in "chunk" (frammenti) più piccoli
Questi chunk vengono trasformati in embedding vettoriali
Quando l'utente pone una domanda, il sistema:

- Converte la domanda in un embedding simile
- Recupera i chunk più rilevanti mediante ricerca di similarità vettoriale
- Fornisce questi chunk come contesto al modello linguistico
- Genera una risposta informata esclusivamente basata sul contesto fornito

Il sistema è progettato specificamente per utilizzare solo le informazioni contenute nei documenti caricati, non la conoscenza generale del modello. Questo garantisce che le risposte siano strettamente pertinenti ai dati dell'utente.

## 🛠️ Tecnologie Utilizzate

- Python - Linguaggio di programmazione principale
- Streamlit - Framework per l'interfaccia utente
- LangChain - Framework per la costruzione di applicazioni basate su LLM
- OpenAI API - Per i modelli GPT e per la creazione di embedding vettoriali
- Chroma DB - Database vettoriale per l'archiviazione degli embedding
- PyPDF, Docx2txt - Per l'elaborazione dei documenti
- RecursiveCharacterTextSplitter - Per il chunking intelligente dei documenti

## 📦 Componenti dell'Applicazione

### Document Q&A

La funzionalità Document Q&A consente agli utenti di interagire con i propri documenti attraverso:

- Caricamento documenti: Supporto per formati PDF, DOCX e TXT
- Elaborazione intelligente: Chunking ottimizzato con sovrapposizione per mantenere il contesto
- Domande contestuali: Risposte generate con riferimento specifico al contenuto del documento
- Generazione di riassunti: Possibilità di ottenere sintesi complete del documento
- Cronologia delle chat: Registrazione delle conversazioni precedenti

#### Come funziona:

- Il documento viene caricato e suddiviso in chunk ottimizzati
- Ogni chunk viene trasformato in un embedding vettoriale
- Quando l'utente pone una domanda, il sistema recupera i chunk più rilevanti
- I chunk vengono utilizzati come contesto per generare una risposta precisa

### Meal Planner

Il Meal Planner intelligente offre:

- Gestione ricette: Caricamento e gestione di ricette in formato JSON
- Ricerca semantica: Trova ricette simili basate su ingredienti o stili di cucina
- Q&A sulle ricette: Risposte a domande specifiche sulle ricette caricate
- Pianificazione pasti: Generazione di piani alimentari personalizzati basati su:
  - Durata (1 giorno, 3 giorni, 1 settimana)
  - Pasti per giorno (colazione, pranzo, cena, spuntini)
  - Preferenze dietetiche (vegetariano, vegano, low-carb, ecc.)
  - Allergie e restrizioni
- Generazione lista della spesa: Creazione automatica di liste della spesa organizzate per categoria

#### Come funziona:

- Le ricette vengono caricate in formato JSON
- Vengono elaborate e trasformate in embedding vettoriali
- L'utente specifica le proprie preferenze e restrizioni
- Il sistema genera un piano pasti personalizzato utilizzando le ricette disponibili
- Opzionalmente, genera una lista della spesa ottimizzata

## 🎓 Percorso Formativo

Questo progetto è il risultato dell'apprendimento e dell'integrazione di conoscenze acquisite attraverso:

- Il corso "Developing LLM Apps with LangChain" della scuola Zero to Mastery.
- Le lezioni sul RAG tenute da Ilyas Chaoua durante il bootcamp AI Engineer di Edgemony.

L'applicazione dimostra l'applicazione pratica di questi concetti in un prodotto funzionale che affronta problemi reali utilizzando tecniche di intelligenza artificiale all'avanguardia.

## 👩‍💻 Autore

Veronica Schembri

Junior AI Engineer | Front-end Developer

- [Website](https://www.veronicaschembri.com/)
- [GitHub](https://github.com/Pandagan-85)
- [LinkedIn](https://www.linkedin.com/in/veronicaschembri/)
