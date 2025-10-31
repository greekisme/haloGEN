# Mira AI Symbolic Logic Engine
# Lightweight, extensible symbolic reasoning engine for IF-THEN style logic rules.
# Provides safe evaluation without code injection vulnerabilities.

import ast
import json
import os
import time
import logging
from typing import Dict, List, Optional, Any, Union

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class Rule:
    """Represents a single IF-THEN logic rule with conditions and actions."""
    
    def __init__(self, rule_id: str, conditions: List[str], actions: List[str], description: Optional[str] = None):
        self.id = rule_id
        self.conditions = conditions
        self.actions = actions
        self.description = description or f"Rule {rule_id}"
    
    def __repr__(self) -> str:
        return f"Rule(id='{self.id}', conditions={self.conditions}, actions={self.actions})"
    
    def __str__(self) -> str:
        conditions_str = " AND ".join(self.conditions)
        actions_str = "; ".join(self.actions)
        return f"IF {conditions_str} THEN {actions_str}"
    
    def matches(self, context: Dict[str, Any]) -> bool:
        """Safely evaluate all conditions against the context."""
        try:
            for condition in self.conditions:
                if not self._evaluate_condition(condition, context):
                    return False
            return True
        except Exception as e:
            logger.warning(f"Error evaluating rule {self.id}: {e}")
            return False
    
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all actions and return updated context."""
        updated_context = context.copy()
        try:
            for action in self.actions:
                self._execute_action(action, updated_context)
            return updated_context
        except Exception as e:
            logger.warning(f"Error applying rule {self.id}: {e}")
            return context
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a single condition using AST parsing."""
        try:
            # Parse the condition into an AST
            tree = ast.parse(condition, mode='eval')
            
            # Evaluate the AST safely
            return self._safe_eval(tree.body, context)
        except Exception as e:
            logger.debug(f"Error parsing condition '{condition}': {e}")
            return False
    
    def _safe_eval(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """Safely evaluate AST nodes with only allowed operations."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return context.get(node.id)
        elif isinstance(node, ast.Attribute):
            obj = self._safe_eval(node.value, context)
            if isinstance(obj, dict):
                return obj.get(node.attr)
            return getattr(obj, node.attr, None)
        elif isinstance(node, ast.Subscript):
            obj = self._safe_eval(node.value, context)
            key = self._safe_eval(node.slice, context)
            if isinstance(obj, (dict, list)):
                return obj[key] if key in obj else None
            return None
        elif isinstance(node, ast.Compare):
            left = self._safe_eval(node.left, context)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._safe_eval(comparator, context)
                if not self._compare_values(left, op, right):
                    return False
            return True
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._safe_eval(child, context) for child in node.values)
            elif isinstance(node.op, ast.Or):
                return any(self._safe_eval(child, context) for child in node.values)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not self._safe_eval(node.operand, context)
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left, context)
            right = self._safe_eval(node.right, context)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right if right != 0 else 0
        elif isinstance(node, ast.Index):
            return self._safe_eval(node.value, context)
        
        return None
    
    def _compare_values(self, left: Any, op: ast.cmpop, right: Any) -> bool:
        """Compare two values using the specified operator."""
        try:
            if isinstance(op, ast.Eq):
                return left == right
            elif isinstance(op, ast.NotEq):
                return left != right
            elif isinstance(op, ast.Lt):
                return left < right
            elif isinstance(op, ast.LtE):
                return left <= right
            elif isinstance(op, ast.Gt):
                return left > right
            elif isinstance(op, ast.GtE):
                return left >= right
            elif isinstance(op, ast.In):
                return left in right if hasattr(right, '__contains__') else False
            elif isinstance(op, ast.NotIn):
                return left not in right if hasattr(right, '__contains__') else True
        except Exception:
            return False
        return False
    
    def _execute_action(self, action: str, context: Dict[str, Any]) -> None:
        """Execute a single action (assignment) safely."""
        try:
            # Parse the action as an assignment
            tree = ast.parse(action, mode='exec')
            
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
                raise ValueError("Action must be a simple assignment")
            
            assign = tree.body[0]
            if len(assign.targets) != 1:
                raise ValueError("Action must have exactly one target")
            
            target = assign.targets[0]
            value = self._safe_eval(assign.value, context)
            
            # Set the value in the context
            self._set_nested_value(context, target, value)
            
        except Exception as e:
            logger.warning(f"Error executing action '{action}': {e}")
    
    def _set_nested_value(self, context: Dict[str, Any], target: ast.AST, value: Any) -> None:
        """Set a value in nested dictionary structure."""
        if isinstance(target, ast.Name):
            context[target.id] = value
        elif isinstance(target, ast.Attribute):
            obj = self._get_nested_object(context, target.value)
            if isinstance(obj, dict):
                obj[target.attr] = value
            else:
                setattr(obj, target.attr, value)
        elif isinstance(target, ast.Subscript):
            obj = self._get_nested_object(context, target.value)
            key = self._safe_eval(target.slice, context)
            if isinstance(obj, dict):
                obj[key] = value
            elif isinstance(obj, list) and isinstance(key, int):
                if key < len(obj):
                    obj[key] = value
                else:
                    obj.extend([None] * (key - len(obj) + 1))
                    obj[key] = value
    
    def _get_nested_object(self, context: Dict[str, Any], node: ast.AST) -> Any:
        """Get nested object from context following the path."""
        if isinstance(node, ast.Name):
            return context.get(node.id, {})
        elif isinstance(node, ast.Attribute):
            obj = self._get_nested_object(context, node.value)
            if isinstance(obj, dict):
                return obj.get(node.attr, {})
            return getattr(obj, node.attr, {})
        elif isinstance(node, ast.Subscript):
            obj = self._get_nested_object(context, node.value)
            key = self._safe_eval(target.slice, context)
            if isinstance(obj, dict):
                return obj.get(key, {})
            elif isinstance(obj, list) and isinstance(key, int) and 0 <= key < len(obj):
                return obj[key]
            return {}
        return {}

class RuleEngine:
    """Core rule engine for managing and evaluating logic rules."""
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self._load_default_rules()
    
    def register_rule(self, rule: Rule) -> None:
        """Register a new rule."""
        self.rules[rule.id] = rule
        logger.info(f"Registered rule: {rule.id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed rule: {rule_id}")
            return True
        return False
    
    def list_rules(self) -> List[Rule]:
        """Get all registered rules."""
        return list(self.rules.values())
    
    def get_active_rules(self, context: Dict[str, Any]) -> List[Rule]:
        """Get rules that match the current context without applying actions."""
        active_rules = []
        for rule in self.rules.values():
            if rule.matches(context):
                active_rules.append(rule)
        return active_rules
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all rules and return updated context."""
        start_time = time.time()
        updated_context = context.copy()
        matched_rules = []
        triggered_rules = []
        
        for rule in self.rules.values():
            if rule.matches(updated_context):
                logger.info(f"[RuleEngine] Matched rule: {rule.id} -> {rule.actions}")
                updated_context = rule.apply(updated_context)
                matched_rules.append(rule)
                triggered_rules.append(rule.id)
        
        # Store triggered rules in context for introspection
        updated_context['rules_triggered'] = triggered_rules
        
        if matched_rules:
            changed_keys = [k for k in updated_context if k not in context or updated_context[k] != context[k]]
            logger.info(f"[RuleEngine] Modified keys: {changed_keys}")
        
        evaluation_time = time.time() - start_time
        logger.info(f"[RuleEngine] Evaluation complete: {len(matched_rules)} rules matched in {evaluation_time:.3f}s")
        
        return updated_context
    
    def explain(self, context: Dict[str, Any]) -> List[str]:
        """Get explanations for all matching rules."""
        explanations = []
        
        for rule in self.rules.values():
            if rule.matches(context):
                explanations.append(f"Rule '{rule.id}': {rule.description}")
        
        return explanations
    
    def save_rules(self, file_path: str = "./data/rules.json") -> None:
        """Save all rules to a JSON file with UTF-8 encoding and preserved order."""
        self._ensure_data_dir()
        
        rules_data = {}
        for rule_id, rule in self.rules.items():
            rules_data[rule_id] = {
                "conditions": rule.conditions,
                "actions": rule.actions,
                "description": rule.description
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False, sort_keys=False)
        
        logger.info(f"Saved {len(self.rules)} rules to {file_path}")
    
    def load_rules(self, file_path: str = "./data/rules.json") -> None:
        """Load rules from a JSON file."""
        if not os.path.exists(file_path):
            logger.info(f"Rules file {file_path} not found, using default rules")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            self.rules.clear()
            for rule_id, rule_data in rules_data.items():
                rule = Rule(
                    rule_id=rule_id,
                    conditions=rule_data.get("conditions", []),
                    actions=rule_data.get("actions", []),
                    description=rule_data.get("description", f"Rule {rule_id}")
                )
                self.rules[rule_id] = rule
            
            logger.info(f"Loaded {len(self.rules)} rules from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading rules from {file_path}: {e}")
    
    def _ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        data_dir = os.path.dirname("./data/rules.json")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            logger.debug(f"Created data directory: {data_dir}")
    
    def _load_default_rules(self) -> None:
        """Load default rules if no rules file exists."""
        default_rules = [
            Rule(
                rule_id="motivate_study",
                conditions=["user.skip_study == True"],
                actions=["ai_response = 'Skipping study might hurt your progress. Want me to help you plan a session?'"],
                description="Encourage user to study when they want to skip"
            ),
            Rule(
                rule_id="encourage_rest",
                conditions=["user.tired == True"],
                actions=["ai_response = 'You seem tired. Taking a short break might help you recharge.'"],
                description="Suggest rest when user is tired"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
        
        logger.info(f"Loaded {len(default_rules)} default rules")

# Global rule engine instance
rule_engine = RuleEngine()

def infer(context: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to evaluate rules and return updated context."""
    return rule_engine.evaluate(context)

# Self-test section
if __name__ == "__main__":
    print("=== Mira AI Logic Engine Self-Test ===")
    
    # Test 1: Rule creation and matching
    print("1. Testing rule creation and matching...")
    test_rule = Rule(
        rule_id="test_rule",
        conditions=["user.mood == 'happy'"],
        actions=["ai_response = 'Great to see you happy!'"],
        description="Test rule for happy mood"
    )
    rule_engine.register_rule(test_rule)
    print("   ✓ Rule created and registered")
    
    # Test 2: Rule evaluation
    print("2. Testing rule evaluation...")
    context = {
        "user": {"mood": "happy", "input": "I feel great!"},
        "ai_response": ""
    }
    
    result = rule_engine.evaluate(context)
    assert result.get("ai_response") == "Great to see you happy!"
    print("   ✓ Rule evaluation working")
    
    # Test 3: Active rules detection
    print("3. Testing active rules detection...")
    active_rules = rule_engine.get_active_rules(context)
    assert len(active_rules) > 0
    print("   ✓ Active rules detection working")
    
    # Test 4: Rule explanations
    print("4. Testing rule explanations...")
    explanations = rule_engine.explain(context)
    assert len(explanations) > 0
    print("   ✓ Rule explanations working")
    
    # Test 5: Rule persistence
    print("5. Testing rule persistence...")
    rule_engine.save_rules()
    print("   ✓ Rules saved successfully")
    
    # Test 6: Nested attribute handling
    print("6. Testing nested attribute handling...")
    nested_context = {
        "user": {"profile": {"mood": "sad"}},
        "ai_response": ""
    }
    
    nested_rule = Rule(
        rule_id="nested_test",
        conditions=["user.profile.mood == 'sad'"],
        actions=["ai_response = 'I understand you are feeling sad.'"],
        description="Test nested attribute access"
    )
    rule_engine.register_rule(nested_rule)
    
    nested_result = rule_engine.evaluate(nested_context)
    assert "sad" in nested_result.get("ai_response", "")
    print("   ✓ Nested attribute handling working")
    
    # Cleanup
    rule_engine.remove_rule("test_rule")
    rule_engine.remove_rule("nested_test")
    
    print("\n=== All Tests Passed! ===")
    print("Logic engine is ready for Mira AI!")

    # Additional test: Clear rules after tests
    rule_engine.rules.clear()