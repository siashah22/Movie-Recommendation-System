import speech_recognition as sr
import pyttsx3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

engine = pyttsx3.init()
def speak(text):
    print("Bot:", text)
    engine.say(text)
    engine.runAndWait()

# Load MovieLens movies dataset
df = pd.read_csv(r"C:\Users\Siya\.kaggle\kaggle.json\BollywoodMovieDetail.csv")  # adjust path if needed
df['description'] = df['genre'].str.replace('|', ' ',regex=False)  # use genres as a basic description
df = df[['title', 'description']]
df['description'] = df['description'].fillna('')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation Function
def get_recommendation(title):
    title=title.lower()
    matches=df[df['title'].str.lower().str.contains(title)]
    if matches.empty:
        return []
    idx=matches.index[0]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)[1:6]
    movie_indices=[i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices].tolist()

# Voice Input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source,duration=0.5)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand. Try again.")
            return None
        except sr.RequestError:
            speak("Could not request results; check your internet connection.")
            return None
# Main Chatbot Loop
def chatbot():
    speak("Hi! Tell me a movie you like, and I’ll recommend something similar.")
    while True:
        user_input = get_voice_input()
        if user_input:
            recommendations = get_recommendation(user_input)
            if recommendations:
                speak(f"If you liked {user_input}, you might also like: " + ", ".join(recommendations))
            else:
                speak("Sorry, I don't know that movie. Try another one.")
        while True:
            speak("Do you want another recommendation? Please say yes or no.")
            response = get_voice_input()
            
            if response:
                response = response.lower().strip()
                print("Recognized response:", response)

                if any(word in response for word in ['no', 'nope', 'nah', 'not really']):
                    speak("Okay, goodbye!")
                    return  # end the chatbot loop
                elif any(word in response for word in ['yes', 'yeah', 'yup', 'sure']):
                    break  # go back to outer loop to ask for another movie
                else:
                    speak("I heard: " + response + ". Please say yes or no.")
            else:
                speak("Didn't get that. Please say yes or no.")


# Run it!
chatbot()
