from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import time
from enum import Enum
import json
import logging

from llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vera - Magicpin AI Agent")
START_TIME = time.time()

llm_client = LLMClient()

class VersionedEntity(BaseModel):
    id: str
    version: int
    payload: dict = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"

class Category(VersionedEntity):
    pass

class Merchant(VersionedEntity):
    pass

class Customer(VersionedEntity):
    pass

class Trigger(VersionedEntity):
    pass

class ConversationState(str, Enum):
    DISCOVERY = "DISCOVERY"
    QUALIFICATION = "QUALIFICATION"
    INTERESTED = "INTERESTED"
    READY = "READY"
    ACTION = "ACTION"
    COMPLETED = "COMPLETED"
    ENDED = "ENDED"

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_auto_reply: bool = False

class Conversation(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    state: ConversationState = ConversationState.DISCOVERY
    history: List[Message] = []
    last_cta: Optional[str] = None
    auto_reply_count: int = 0

class StateStore:
    def __init__(self):
        self.categories: Dict[str, Category] = {}
        self.merchants: Dict[str, Merchant] = {}
        self.customers: Dict[str, Customer] = {}
        self.triggers: Dict[str, Trigger] = {}
        self.conversations: Dict[str, Conversation] = {}

    def get_conversation(self, merchant_id: str, customer_id: Optional[str] = None) -> Conversation:
        # Create a deterministic conversation ID
        conv_id = f"{merchant_id}_{customer_id or 'merchant_only'}"
        if conv_id not in self.conversations:
            self.conversations[conv_id] = Conversation(
                conversation_id=conv_id,
                merchant_id=merchant_id,
                customer_id=customer_id
            )
        return self.conversations[conv_id]

store = StateStore()

class ContextPayload(BaseModel):
    scope: str # 'category', 'merchant', 'customer', 'trigger'
    context_id: str
    version: int
    payload: dict
    delivered_at: str

class TickPayload(BaseModel):
    now: str
    available_triggers: List[str] = []
    
class ReplyPayload(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.get("/v1/healthz")
def healthz():
    counts = {
        "category": len(store.categories),
        "merchant": len(store.merchants),
        "customer": len(store.customers),
        "trigger": len(store.triggers)
    }
    return {
        "status": "ok", 
        "uptime_seconds": int(time.time() - START_TIME), 
        "contexts_loaded": counts
    }

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Alpha",
        "team_members": ["Agent"],
        "model": "mistral-small-latest",
        "approach": "In-Memory State + Opportunity Engine + Mistral Composer",
        "contact_email": "bot@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/v1/context")
async def receive_context(payload: ContextPayload):
    scope = payload.scope
    entity_id = payload.context_id
    version = payload.version
    data = payload.payload

    if scope == "category":
        existing = store.categories.get(entity_id)
        if existing and version <= existing.version:
            return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": existing.version})
        store.categories[entity_id] = Category(id=entity_id, version=version, payload=data)
        
    elif scope == "merchant":
        existing = store.merchants.get(entity_id)
        if existing and version <= existing.version:
            return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": existing.version})
        store.merchants[entity_id] = Merchant(id=entity_id, version=version, payload=data)
        
    elif scope == "customer":
        existing = store.customers.get(entity_id)
        if existing and version <= existing.version:
            return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": existing.version})
        store.customers[entity_id] = Customer(id=entity_id, version=version, payload=data)
        
    elif scope == "trigger":
        existing = store.triggers.get(entity_id)
        if existing and version <= existing.version:
            return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": existing.version})
        store.triggers[entity_id] = Trigger(id=entity_id, version=version, payload=data)
    else:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope", "details": scope})
        
    return {
        "accepted": True, 
        "ack_id": f"ack_{entity_id}_v{version}", 
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/v1/tick")
def handle_tick(payload: TickPayload):
    actions = []
    
    active_triggers = []
    for trg_id in payload.available_triggers:
        trg = store.triggers.get(trg_id)
        if trg and trg.payload.get("status", "ACTIVE") == "ACTIVE":
            active_triggers.append(trg)
            
    candidates = []
    
    for trigger in active_triggers:
        merchant_id = trigger.payload.get("merchant_id")
        merchant = store.merchants.get(merchant_id)
        if not merchant:
            continue
            
        category_slug = merchant.payload.get("category_slug") or merchant.payload.get("category_id") or merchant.payload.get("category")
        category = store.categories.get(category_slug)
        customer_id = trigger.payload.get("customer_id")
        customer = store.customers.get(customer_id) if customer_id else None
        
        conv = store.get_conversation(merchant_id, customer_id)
        if conv.state in [ConversationState.INTERESTED, ConversationState.READY, ConversationState.QUALIFICATION]:
            continue
            
        trg_type = trigger.payload.get("type", "unknown")
        relevance = 80 
        business_impact = 90 if "drop" in trg_type else 60
        urgency = 85 if "drop" in trg_type else 50
        freshness = 100 
        repetition_risk = 0 
        
        score = (0.3 * relevance) + (0.25 * business_impact) + (0.15 * urgency) + (0.15 * freshness) - (0.2 * repetition_risk)
        trigger.payload["score"] = score
        
        if score > 50:
            candidates.append((score, trigger, merchant, category, customer, conv))
            
    merchant_top_triggers = {}
    for score, t, m, cat, cus, conv in candidates:
        if m.id not in merchant_top_triggers or score > merchant_top_triggers[m.id][0]:
            merchant_top_triggers[m.id] = (score, t, m, cat, cus, conv)
            
    top_candidates = sorted(merchant_top_triggers.values(), key=lambda x: x[0], reverse=True)[:20]
    
    for score, trigger, merchant, category, customer, conv in top_candidates:
        trg_type = trigger.payload.get("type", "unknown")
        strategy = {
            "recipient": "merchant" if not customer else "customer",
            "objective": f"Address {trg_type}",
            "trigger": trg_type,
            "evidence": trigger.payload.get("payload", trigger.payload),
            "recommended_action": "Review the suggested improvements",
            "tone": category.payload.get("voice", {}).get("tone", "simple_english") if category else "simple_english",
            "language": merchant.payload.get("identity", {}).get("languages", ["en"])[0] if merchant else "en"
        }
        
        context_bundle = {
            "merchant": merchant.payload,
            "category": category.payload if category else None,
            "customer": customer.payload if customer else None
        }
        
        generated_message = llm_client.generate_message(strategy, context_bundle)
        
        if not generated_message:
            continue
            
        trigger.payload["status"] = "HANDLED"
        conv.history.append(Message(role="agent", content=generated_message))
        conv.state = ConversationState.DISCOVERY
        
        actions.append({
            "conversation_id": conv.conversation_id,
            "merchant_id": merchant.id,
            "customer_id": customer.id if customer else None,
            "send_as": "vera",
            "trigger_id": trigger.id,
            "template_name": "vera_dynamic_composer",
            "template_params": [],
            "body": generated_message,
            "cta": "open_ended",
            "suppression_key": trigger.payload.get("suppression_key", "default"),
            "rationale": f"High score ({score:.1f}) for trigger {trg_type}"
        })

    return {"actions": actions}

@app.post("/v1/reply")
def handle_reply(payload: ReplyPayload):
    merchant_id = payload.merchant_id
    customer_id = payload.customer_id
    message_text = payload.message
    
    if not merchant_id:
        return {"action": "wait", "rationale": "Missing merchant_id in reply"}
        
    conv = store.get_conversation(merchant_id, customer_id)
    
    is_auto_reply = "thank you for contacting" in message_text.lower() or "get back to you" in message_text.lower()
    
    if is_auto_reply:
        conv.auto_reply_count += 1
        if conv.auto_reply_count >= 2:
            conv.state = ConversationState.ENDED
            return {"action": "end", "rationale": "Auto-reply hell detected. Ending conversation."}
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Suspected auto-reply. Backing off."}
        
    conv.auto_reply_count = 0 
    conv.history.append(Message(role="merchant", content=message_text))
    
    history_dicts = [{"role": m.role, "content": m.content} for m in conv.history]
    intent_data = llm_client.classify_intent(history_dicts, message_text)
    intent = intent_data["intent"]
    logger.info(f"Classified intent: {intent} for message: {message_text}")
    
    if intent == "NOT_INTERESTED":
        conv.state = ConversationState.ENDED
        return {"action": "end", "rationale": "Merchant explicitly not interested."}
        
    elif intent == "OFF_TOPIC":
        redirect_msg = "I'm here to help with your Magicpin listing and business performance. Returning to that, would you like to continue our previous discussion?"
        conv.history.append(Message(role="agent", content=redirect_msg))
        return {"action": "send", "body": redirect_msg, "cta": "open_ended", "rationale": "Polite redirection from off-topic query."}
        
    elif intent == "CONFUSED":
        clarify_msg = "Let me simplify: I saw a drop in your performance metrics, and I have a few simple steps we can take to fix it. Sound good?"
        conv.history.append(Message(role="agent", content=clarify_msg))
        return {"action": "send", "body": clarify_msg, "cta": "yes_no", "rationale": "Simplifying the value proposition for confused merchant."}
        
    elif intent == "READY" or intent == "INTERESTED":
        if conv.state == ConversationState.READY or intent == "READY":
            conv.state = ConversationState.ACTION
            action_msg = "Great, let's take action! I have applied the changes to your listing."
            conv.history.append(Message(role="agent", content=action_msg))
            return {"action": "send", "body": action_msg, "cta": "none", "rationale": "Transitioned to Action state following merchant readiness."}
        else:
            conv.state = ConversationState.READY
            ready_msg = "Excellent. I can apply these changes for you right now. Should I proceed?"
            conv.history.append(Message(role="agent", content=ready_msg))
            return {"action": "send", "body": ready_msg, "cta": "yes_no", "rationale": "Advanced to Ready state after merchant showed interest."}

    # Fallback
    return {"action": "wait", "wait_seconds": 3600, "rationale": "Uncertain intent, backing off."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
