import os
from dotenv import load_dotenv

load_dotenv()
import json
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RaceLensAnswer:
    answer: str
    sources: List[str]
    confidence: float

class GeminiRaceAnalyzer:
    """Uses Google Gemini (FREE API)"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            print("⚠ GOOGLE_API_KEY not set")
            print("Get free key: https://ai.google.dev")
            self.enabled = False
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('models/gemini-2.5-flash')
                self.enabled = True
                print("✓ Gemini API connected")
            except ImportError:
                print("⚠ Install: pip install google-generativeai")
                self.enabled = False
    
    def answer_question(self, question: str) -> RaceLensAnswer:
        """Use Gemini to answer ANY F1 question"""
        
        if not self.enabled:
            return self._fallback(question)
        
        try:
            prompt = f"""You are an expert Formula 1 race analyst and strategist. 
Answer this question about F1 racing with specific facts, examples, and explanations.

QUESTION: {question}

Provide:
1. Direct answer to the question
2. Specific F1 facts and examples
3. Why this matters in racing
4. Relevant regulations or rules if applicable

Be detailed, technical but understandable."""
            
            response = self.model.generate_content(prompt)
            answer_text = response.text
            
            return RaceLensAnswer(
                answer=answer_text,
                sources=["Google Gemini AI Analysis", "F1 Knowledge", "Racing Strategy"],
                confidence=85
            )
        
        except Exception as e:
            print(f"Gemini error: {e}")
            return self._fallback(question)
    
    @staticmethod
    def _fallback(question: str) -> RaceLensAnswer:
        """Fallback if API fails"""
        return RaceLensAnswer(
            answer=f"Unable to reach Gemini API. Your question was: {question}",
            sources=["Offline mode"],
            confidence=0
        )

class RaceLensOrchestrator:
    def __init__(self):
        self.analyzer = GeminiRaceAnalyzer()
    
    def ask_the_race(self, question: str) -> RaceLensAnswer:
        """Answer any F1 question using Gemini"""
        return self.analyzer.answer_question(question)

# Demo
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏁 RaceLens AI - Gemini Powered")
    print("="*70)
    
    orchestrator = RaceLensOrchestrator()
    
    questions = [
        "Why do F1 cars use three different tire compounds?",
        "Explain the undercut strategy in pit stop timing",
        "How does DRS work and why is it important?",
        "What happens to tire performance in wet weather?",
        "How do teams calculate the optimal pit window?",
        "Why is fuel management critical in modern F1?",
        "What's the difference between qualifying and race pace?",
        "How do drivers manage tire temperature in corners?"
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n🏎️ Question {i}: {q}")
        print("-" * 70)
        answer = orchestrator.ask_the_race(q)
        print(f"Answer:\n{answer.answer[:500]}...\n")
        print(f"Confidence: {answer.confidence}%")