import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RaceLensAnswer:
    answer: str
    sources: List[str]
    confidence: float
    reasoning_steps: List[str] = None
    counterfactuals: Optional[List[Dict]] = None

class GraniteReasoner:
    def __init__(self):
        self.enabled = False
        api_token = os.getenv("REPLICATE_API_TOKEN", "")
        if api_token:
            try:
                import importlib
                # dynamically import to avoid static import resolution issues
                ChatReplicate = importlib.import_module("langchain_replicate").ChatReplicate
                self.model = ChatReplicate(
                    model="ibm-granite/granite-3.1-8b-instruct",
                    replicate_api_token=api_token,
                    model_kwargs={"max_tokens": 1024, "temperature": 0.3}
                )
                self.enabled = True
                print("✓ Granite connected")
            except:
                print("⚠ Using demo responses")
        else:
            print("⚠ No API token - using demo responses")
    
    def analyze(self, question: str, context: Dict) -> str:
        if self.enabled:
            try:
                prompt = f"You are an F1 race analyst.\n\nQuestion: {question}\n\nProvide clear answer."
                return self.model.invoke(prompt).content
            except:
                return self._demo(question)
        return self._demo(question)
    
    @staticmethod
    def _demo(question: str) -> str:
        q = question.lower()
        if "pit" in q or "lose" in q:
            return """ANSWER: Norris lost the lead due to failed undercut. He pitted on lap 32 while Verstappen stayed out. His outlap was 1.3s slower due to cool 18°C track temp—hard tires didn't warm up fast enough.

SOURCES: FIA Sporting Regulations Article 34.7-34.8, Real-time telemetry

CONFIDENCE: 89%"""
        elif "tire" in q or "new" in q:
            return """ANSWER: F1 uses three tire compounds:
- SOFT (red): Max grip, 18-22 laps
- MEDIUM (yellow): Balanced, 22-28 laps  
- HARD (white): Lower grip, 35-45 laps

Teams must use at least TWO different compounds.

SOURCES: FIA Technical Regulations Article 10

CONFIDENCE: 98%"""
        else:
            return """ANSWER: That's a great racing question. Modern F1 strategy balances tire degradation, fuel consumption, and track position. Teams pit when tires degrade or competitors pit (undercut/overcut).

SOURCES: FIA Regulations, Real-time telemetry

CONFIDENCE: 78%"""

class F1DataFetcher:
    @staticmethod
    def get_sample() -> Dict:
        return {
            "circuit": "Bahrain",
            "lap": 35,
            "total_laps": 57,
            "positions": [
                {"position": 1, "driver": "Lando Norris", "gap": "—", "tire": "HARD", "speed": 289.5},
                {"position": 2, "driver": "Max Verstappen", "gap": "+2.3s", "tire": "HARD", "speed": 288.2},
                {"position": 3, "driver": "Charles Leclerc", "gap": "+5.8s", "tire": "MEDIUM", "speed": 287.1},
            ]
        }

class RegulationParser:
    REGULATIONS = {
        "pit_stops": "Minimum 2 second stop, 80km/h pit lane limit",
        "tire_compounds": {
            "soft": "18-22 laps",
            "medium": "22-28 laps",
            "hard": "35-45 laps"
        }
    }

class RaceLensOrchestrator:
    def __init__(self):
        self.reasoner = GraniteReasoner()
        self.data = F1DataFetcher()
    
    def ask_the_race(self, question: str) -> RaceLensAnswer:
        race_data = self.data.get_sample()
        context = {"current_race": race_data}
        answer_text = self.reasoner.analyze(question, context)
        
        sources = self._get_sources(question)
        confidence = self._extract_confidence(answer_text)
        
        return RaceLensAnswer(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            reasoning_steps=[]
        )
    
    @staticmethod
    def _extract_confidence(text: str) -> int:
        try:
            if "CONFIDENCE:" in text:
                conf = text.split("CONFIDENCE:")[1].split("%")[0].strip()
                return int(conf)
        except:
            pass
        return 85
    
    @staticmethod
    def _get_sources(question: str) -> List[str]:
        q = question.lower()
        if "pit" in q:
            return ["FIA Sporting Regulations Article 34.7-34.8", "Real-time telemetry"]
        elif "tire" in q:
            return ["FIA Technical Regulations Article 10", "Pirelli tire guide"]
        return ["FIA Regulations", "Session telemetry"]

def run_demo():
    print("\n" + "="*70)
    print("🏁 RaceLens AI - Demo")
    print("="*70)
    
    racelens = RaceLensOrchestrator()
    
    # Demo 1
    print("\n📊 DEMO 1: Strategy")
    print("-" * 70)
    q1 = "Why did Norris just lose the lead?"
    a1 = racelens.ask_the_race(q1)
    print(f"Q: {q1}\nA: {a1.answer}")
    print(f"Confidence: {a1.confidence}%")
    print(f"Sources: {a1.sources[0]}")
    
    # Demo 2
    print("\n" + "="*70)
    print("\n👶 DEMO 2: Rookie Mode")
    print("-" * 70)
    q2 = "Why do they use different colored tires?"
    a2 = racelens.ask_the_race(q2)
    print(f"Q: {q2}\nA: {a2.answer}")
    print(f"Confidence: {a2.confidence}%")
    
    # Demo 3
    print("\n" + "="*70)
    print("\n📺 DEMO 3: Broadcaster")
    print("-" * 70)
    q3 = "Last 5 wet races at Spa - pole vs winner?"
    a3 = racelens.ask_the_race(q3)
    print(f"Q: {q3}\nA: {a3.answer}")
    print(f"Confidence: {a3.confidence}%")
    
    print("\n" + "="*70)
    print("✅ Done!")
    print("="*70)

if __name__ == "__main__":
    run_demo()