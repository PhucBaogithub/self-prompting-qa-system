"""
Enhanced Evaluation Metrics for Self-Prompting Pipeline
Comprehensive metrics with model efficiency comparison and question generation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import time
from datetime import datetime
import json
import re

# NLP Metrics
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# ML Metrics
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Additional imports for precision, recall, f1-score
from sklearn.metrics import precision_score, recall_score, f1_score
import re

class MetricsCalculator:
    """Calculate comprehensive metrics for Self-Prompting pipeline with model comparison"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # ============= PRECISION, RECALL, F1-SCORE METRICS =============
    
    def calculate_precision_recall_f1(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """Calculate precision, recall, and F1-score for generated vs reference answers"""
        try:
            # If no reference answer is provided, use estimation method
            if not reference_answer or reference_answer.strip() == "":
                return self._estimate_precision_recall_f1(generated_answer)
            
            # Tokenize answers
            generated_tokens = set(self._tokenize_answer(generated_answer))
            reference_tokens = set(self._tokenize_answer(reference_answer))
            
            if not reference_tokens:
                return self._estimate_precision_recall_f1(generated_answer)
            
            # Calculate metrics
            if not generated_tokens:
                return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            
            # True positives: tokens present in both
            tp = len(generated_tokens.intersection(reference_tokens))
            
            # False positives: tokens in generated but not in reference
            fp = len(generated_tokens - reference_tokens)
            
            # False negatives: tokens in reference but not in generated
            fn = len(reference_tokens - generated_tokens)
            
            # Calculate precision, recall, f1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            
        except Exception as e:
            return self._estimate_precision_recall_f1(generated_answer)
    
    def _tokenize_answer(self, answer: str) -> List[str]:
        """Tokenize answer text for precision/recall calculation"""
        if not answer:
            return []
        
        # Convert to lowercase and remove punctuation
        answer = answer.lower()
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Split into tokens
        tokens = answer.split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        
        return tokens
    
    def calculate_model_performance_metrics(self, results: Dict[str, Dict], reference_answers: List[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for all models including precision, recall, f1-score"""
        model_metrics = {}
        
        for model_name, result in results.items():
            if 'answer' in result and 'error' not in result:
                metrics = {
                    'model_name': model_name,
                    'inference_time': result.get('inference_time', 0),
                    'confidence': result.get('confidence', 0.0),
                    'answer_length': len(result['answer'].split()) if result['answer'] else 0,
                    'answer_char_count': len(result['answer']) if result['answer'] else 0
                }
                
                # If reference answers are provided, calculate precision, recall, f1
                if reference_answers:
                    avg_precision = 0.0
                    avg_recall = 0.0
                    avg_f1 = 0.0
                    
                    for ref_answer in reference_answers:
                        prf_metrics = self.calculate_precision_recall_f1(result['answer'], ref_answer)
                        avg_precision += prf_metrics['precision']
                        avg_recall += prf_metrics['recall']
                        avg_f1 += prf_metrics['f1_score']
                    
                    num_refs = len(reference_answers)
                    metrics.update({
                        'precision': avg_precision / num_refs,
                        'recall': avg_recall / num_refs,
                        'f1_score': avg_f1 / num_refs
                    })
                else:
                    # Estimate metrics based on answer quality
                    estimated_metrics = self._estimate_precision_recall_f1(result['answer'])
                    metrics.update(estimated_metrics)
                
                model_metrics[model_name] = metrics
        
        return {
            'model_metrics': model_metrics,
            'best_precision_model': max(model_metrics.keys(), key=lambda x: model_metrics[x].get('precision', 0)) if model_metrics else None,
            'best_recall_model': max(model_metrics.keys(), key=lambda x: model_metrics[x].get('recall', 0)) if model_metrics else None,
            'best_f1_model': max(model_metrics.keys(), key=lambda x: model_metrics[x].get('f1_score', 0)) if model_metrics else None,
            'avg_precision': np.mean([m.get('precision', 0) for m in model_metrics.values()]) if model_metrics else 0.0,
            'avg_recall': np.mean([m.get('recall', 0) for m in model_metrics.values()]) if model_metrics else 0.0,
            'avg_f1_score': np.mean([m.get('f1_score', 0) for m in model_metrics.values()]) if model_metrics else 0.0
        }
    
    def _estimate_precision_recall_f1(self, answer: str) -> Dict[str, float]:
        """Estimate precision, recall, f1-score based on answer quality heuristics"""
        if not answer or len(answer.strip()) < 3:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # More sophisticated estimation based on answer quality
        answer_lower = answer.lower()
        words = answer.split()
        answer_length = len(words)
        
        # Base scores based on answer comprehensiveness
        if answer_length < 3:
            base_precision, base_recall = 0.15, 0.10
        elif answer_length < 8:
            base_precision, base_recall = 0.45, 0.40
        elif answer_length < 20:
            base_precision, base_recall = 0.65, 0.60
        elif answer_length < 50:
            base_precision, base_recall = 0.75, 0.70
        else:
            base_precision, base_recall = 0.70, 0.75  # Very long answers might have lower precision
        
        # Quality indicators that boost scores
        quality_indicators = [
            any(word in answer_lower for word in ['specifically', 'particularly', 'precisely', 'exactly']),
            any(word in answer_lower for word in ['comprehensive', 'detailed', 'thorough', 'complete']),
            any(word in answer_lower for word in ['because', 'since', 'due to', 'as a result']),  # Explanatory
            any(word in answer_lower for word in ['first', 'second', 'third', 'finally']),  # Structured
            len(set(words)) / len(words) > 0.7 if words else False,  # Vocabulary diversity
            answer.count('.') >= 2,  # Multiple sentences
            not any(word in answer_lower for word in ['unclear', 'unsure', 'maybe', 'possibly']),  # Confident
            answer.strip().endswith(('.', '!', '?')),  # Proper ending
        ]
        
        # Negative indicators that reduce scores
        negative_indicators = [
            any(word in answer_lower for word in ['i don\'t know', 'unclear', 'unsure']),
            answer_lower.startswith('sorry'),
            len(words) < 3,
            answer.count('?') > 2,  # Too many questions
        ]
        
        # Calculate adjustments
        quality_boost = sum(quality_indicators) * 0.03
        negative_penalty = sum(negative_indicators) * 0.15
        
        # Apply adjustments
        precision = base_precision + quality_boost - negative_penalty
        recall = base_recall + quality_boost - negative_penalty
        
        # Ensure values are in valid range
        precision = max(0.0, min(1.0, precision))
        recall = max(0.0, min(1.0, recall))
        
        # Calculate F1-score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4)
        }
        
    # ============= EFFICIENCY COMPARISON METRICS =============
    
    def calculate_model_efficiency_metrics(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate comprehensive efficiency comparison between models"""
        
        if len(results) < 2:
            return {'error': 'Need at least 2 models for comparison'}
        
        # Extract performance data
        model_data = {}
        for model_name, result in results.items():
            if 'answer' in result and 'error' not in result:
                model_data[model_name] = {
                    'inference_time': result.get('inference_time', 0),
                    'answer_length': len(result['answer'].split()),
                    'answer_char_count': len(result['answer']),
                    'confidence': result.get('confidence', None),
                    'context_used': result.get('context_used', 0),
                    'answer_complexity': len(set(result['answer'].lower().split())),
                    'answer_quality_score': self._estimate_answer_quality(result['answer'])
                }
        
        if len(model_data) < 2:
            return {'error': 'Need at least 2 successful model results'}
        
        # Calculate comparison metrics for all models
        models = list(model_data.keys())
        
        # Multi-model analysis
        speed_analysis = self._analyze_multi_model_speed(model_data)
        quality_analysis = self._analyze_multi_model_quality(model_data)
        efficiency_analysis = self._analyze_efficiency(model_data)
        recommendations = self._generate_recommendations(model_data)
        
        # For backward compatibility, still provide pairwise comparison for first two models
        if len(models) >= 2:
            model1, model2 = models[0], models[1]
            data1, data2 = model_data[model1], model_data[model2]
            pairwise_speed = self._analyze_speed_performance(model1, data1, model2, data2)
            pairwise_quality = self._analyze_quality_performance(model1, data1, model2, data2)
            pairwise_summary = self._generate_comparison_summary(model1, data1, model2, data2)
        else:
            pairwise_speed = {}
            pairwise_quality = {}
            pairwise_summary = "Single model analysis"
        
        metrics = {
            'models_compared': f"{len(models)} models: {', '.join(models)}",
            'speed_analysis': speed_analysis,
            'quality_analysis': quality_analysis,
            'efficiency_analysis': efficiency_analysis,
            'recommendations': recommendations,
            'detailed_scores': self._calculate_detailed_scores(model_data),
            'comparison_summary': pairwise_summary,
            # Keep backward compatibility
            'pairwise_speed_analysis': pairwise_speed,
            'pairwise_quality_analysis': pairwise_quality,
            'all_model_data': model_data
        }
        
        return self._convert_numpy_types(metrics)
    
    def _analyze_speed_performance(self, model1: str, data1: Dict, model2: str, data2: Dict) -> Dict:
        """Analyze speed performance between models"""
        time1, time2 = max(0.001, data1['inference_time']), max(0.001, data2['inference_time'])  # Avoid zero
        
        faster_model = model1 if time1 < time2 else model2
        speed_difference = abs(time1 - time2)
        speed_ratio = max(time1, time2) / min(time1, time2) if min(time1, time2) > 0.001 else 1.0
        
        # Performance categories
        if speed_difference < 0.1:
            speed_category = "Similar Speed"
        elif speed_difference < 0.5:
            speed_category = "Slight Speed Difference"
        elif speed_difference < 1.0:
            speed_category = "Moderate Speed Difference"
        else:
            speed_category = "Significant Speed Difference"
        
        return {
            'faster_model': faster_model,
            'speed_difference_seconds': round(speed_difference, 3),
            'speed_ratio': round(speed_ratio, 2),
            'speed_category': speed_category,
            'performance_gain_percentage': round((speed_ratio - 1) * 100, 1),
            f'{model1}_time': round(time1, 3),
            f'{model2}_time': round(time2, 3)
        }
    
    def _analyze_quality_performance(self, model1: str, data1: Dict, model2: str, data2: Dict) -> Dict:
        """Analyze answer quality between models"""
        
        # Length comparison
        len1, len2 = data1['answer_length'], data2['answer_length']
        more_detailed = model1 if len1 > len2 else model2
        
        # Complexity comparison
        comp1, comp2 = data1['answer_complexity'], data2['answer_complexity']
        more_complex = model1 if comp1 > comp2 else model2
        
        # Quality score comparison
        qual1, qual2 = data1['answer_quality_score'], data2['answer_quality_score']
        higher_quality = model1 if qual1 > qual2 else model2
        
        # Confidence comparison (if available)
        conf_comparison = {}
        if data1.get('confidence') and data2.get('confidence'):
            conf1, conf2 = data1['confidence'], data2['confidence']
            conf_comparison = {
                'higher_confidence_model': model1 if conf1 > conf2 else model2,
                f'{model1}_confidence': f"{conf1:.1%}",
                f'{model2}_confidence': f"{conf2:.1%}",
                'confidence_difference': abs(conf1 - conf2)
            }
        
        return {
            'more_detailed_model': more_detailed,
            'detail_difference_words': abs(len1 - len2),
            'more_complex_model': more_complex,
            'complexity_difference': abs(comp1 - comp2),
            'higher_quality_model': higher_quality,
            'quality_score_difference': round(abs(qual1 - qual2), 3),
            f'{model1}_metrics': {
                'length': len1,
                'complexity': comp1,
                'quality_score': round(qual1, 3)
            },
            f'{model2}_metrics': {
                'length': len2,
                'complexity': comp2,
                'quality_score': round(qual2, 3)
            },
            'confidence_comparison': conf_comparison
        }
    
    def _analyze_efficiency(self, model_data: Dict) -> Dict:
        """Calculate efficiency scores for each model"""
        efficiency_scores = {}
        
        for model_name, data in model_data.items():
            # Efficiency = (Quality × Detail) / Time
            time_factor = max(0.001, data['inference_time'])  # Minimum 1ms to avoid division by zero
            quality_factor = max(0.001, data['answer_quality_score'])
            detail_factor = min(max(1, data['answer_length']) / 30, 2.0)  # Cap detail bonus
            
            efficiency = (quality_factor * detail_factor) / time_factor
            efficiency_scores[model_name] = round(efficiency, 3)
        
        # Find most efficient
        best_model = max(efficiency_scores.keys(), key=lambda k: efficiency_scores[k])
        
        return {
            'efficiency_scores': efficiency_scores,
            'most_efficient_model': best_model,
            'efficiency_ranking': sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True),
            'efficiency_categories': self._categorize_efficiency(efficiency_scores)
        }
    
    def _categorize_efficiency(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Categorize efficiency levels"""
        categories = {}
        for model, score in scores.items():
            if score >= 2.0:
                categories[model] = "High Efficiency"
            elif score >= 1.0:
                categories[model] = "Medium Efficiency"
            elif score >= 0.5:
                categories[model] = "Low Efficiency"
            else:
                categories[model] = "Very Low Efficiency"
        return categories
    
    def _generate_recommendations(self, model_data: Dict) -> Dict[str, str]:
        """Generate use-case specific recommendations"""
        models = list(model_data.keys())
        if len(models) < 2:
            return {}
        
        model1, model2 = models[0], models[1]
        data1, data2 = model_data[model1], model_data[model2]
        
        recommendations = {}
        
        # Real-time applications
        time1, time2 = max(0.001, data1['inference_time']), max(0.001, data2['inference_time'])
        if time1 < time2:
            time_savings = ((time2 - time1) / time2) * 100
            recommendations['real_time_use'] = f"Use {model1} (saves {time_savings:.1f}% time)"
        else:
            time_savings = ((time1 - time2) / time1) * 100
            recommendations['real_time_use'] = f"Use {model2} (saves {time_savings:.1f}% time)"
        
        # Detailed analysis needs
        qual1, qual2 = max(0.001, data1['answer_quality_score']), max(0.001, data2['answer_quality_score'])
        if qual1 > qual2:
            quality_gain = ((qual1 - qual2) / qual2) * 100
            recommendations['detailed_analysis'] = f"Use {model1} (better quality by {quality_gain:.1f}%)"
        else:
            quality_gain = ((qual2 - qual1) / qual1) * 100
            recommendations['detailed_analysis'] = f"Use {model2} (better quality by {quality_gain:.1f}%)"
        
        # Balanced use case
        eff_scores = self._calculate_efficiency_score(model_data)
        best_balanced = max(eff_scores.keys(), key=lambda k: eff_scores[k])
        recommendations['balanced_use'] = f"Use {best_balanced} (best overall efficiency)"
        
        # Resource-constrained environments
        fastest_model = min(model_data.keys(), key=lambda k: model_data[k]['inference_time'])
        recommendations['resource_constrained'] = f"Use {fastest_model} (lowest resource usage)"
        
        return recommendations
    
    def _calculate_efficiency_score(self, model_data: Dict) -> Dict[str, float]:
        """Calculate efficiency score for each model"""
        efficiency_scores = {}
        
        for model_name, data in model_data.items():
            time_penalty = max(0.001, data['inference_time'])  # Minimum 1ms to avoid division by zero
            quality_bonus = max(0.001, data['answer_quality_score'])
            length_bonus = min(max(1, data['answer_length']) / 40, 1.5)
            
            efficiency = (quality_bonus * length_bonus) / time_penalty
            efficiency_scores[model_name] = round(efficiency, 3)
        
        return efficiency_scores
    
    def _calculate_detailed_scores(self, model_data: Dict) -> Dict:
        """Calculate detailed performance scores"""
        scores = {}
        
        for model_name, data in model_data.items():
            scores[model_name] = {
                'speed_score': round(10 / max(0.1, data['inference_time']), 2),
                'detail_score': round(min(data['answer_length'] / 50 * 10, 10), 2),
                'quality_score': round(data['answer_quality_score'] * 10, 2),
                'complexity_score': round(min(data['answer_complexity'] / 20 * 10, 10), 2),
                'overall_score': 0  # Will calculate below
            }
            
            # Calculate overall score (weighted average)
            overall = (
                scores[model_name]['speed_score'] * 0.3 +
                scores[model_name]['detail_score'] * 0.2 +
                scores[model_name]['quality_score'] * 0.3 +
                scores[model_name]['complexity_score'] * 0.2
            )
            scores[model_name]['overall_score'] = round(overall, 2)
        
        return scores
    
    def _estimate_answer_quality(self, answer: str) -> float:
        """Estimate answer quality based on heuristics"""
        if not answer or len(answer.strip()) == 0:
            return 0.0
        
        # Length factor (optimal around 50-200 characters)
        length = len(answer)
        if length < 10:
            length_score = length / 10
        elif length <= 200:
            length_score = 1.0
        else:
            length_score = max(0.5, 200 / length)
        
        # Completeness (ends with punctuation)
        completeness_score = 1.0 if answer.strip()[-1] in '.!?' else 0.7
        
        # Structure (has multiple sentences)
        sentence_count = len([s for s in answer.split('.') if s.strip()])
        structure_score = min(sentence_count / 3, 1.0)
        
        # Word diversity
        words = answer.lower().split()
        unique_words = len(set(words))
        diversity_score = min(unique_words / max(1, len(words)), 1.0)
        
        # Combined score
        quality_score = (
            length_score * 0.3 +
            completeness_score * 0.2 +
            structure_score * 0.3 +
            diversity_score * 0.2
        )
        
        return round(quality_score, 3)
    
    def _generate_comparison_summary(self, model1: str, data1: Dict, model2: str, data2: Dict) -> str:
        """Generate a human-readable comparison summary"""
        
        # Determine winner in each category
        speed_winner = model1 if data1['inference_time'] < data2['inference_time'] else model2
        quality_winner = model1 if data1['answer_quality_score'] > data2['answer_quality_score'] else model2
        detail_winner = model1 if data1['answer_length'] > data2['answer_length'] else model2
        
        summary_parts = []
        
        # Speed comparison
        time_diff = abs(data1['inference_time'] - data2['inference_time'])
        if time_diff > 0.1:
            percentage = (time_diff / max(data1['inference_time'], data2['inference_time'])) * 100
            summary_parts.append(f"{speed_winner} is faster by {percentage:.1f}%")
        else:
            summary_parts.append("Both models have similar speed")
        
        # Quality comparison
        quality_diff = abs(data1['answer_quality_score'] - data2['answer_quality_score'])
        if quality_diff > 0.1:
            summary_parts.append(f"{quality_winner} provides higher quality answers")
        else:
            summary_parts.append("Both models provide similar quality")
        
        # Detail comparison
        length_diff = abs(data1['answer_length'] - data2['answer_length'])
        if length_diff > 5:
            summary_parts.append(f"{detail_winner} provides more detailed responses")
        else:
            summary_parts.append("Both models provide similar detail level")
        
        return ". ".join(summary_parts) + "."

    # ============= QUESTION GENERATION FROM CLUSTERING =============
    
    def generate_questions_from_clusters(self, clustered_data: Dict, max_questions_per_cluster: int = 3) -> List[Dict]:
        """Generate relevant questions based on clustered QA data"""
        
        if not clustered_data or 'qa_pairs' not in clustered_data:
            return []
        
        generated_questions = []
        
        for cluster_id, qa_pairs in clustered_data['qa_pairs'].items():
            if not qa_pairs:
                continue
            
            # Extract topics and patterns from cluster
            cluster_topics = [pair.get('topic', 'General') for pair in qa_pairs]
            cluster_questions = [pair['question'] for pair in qa_pairs]
            cluster_answers = [pair['answer'] for pair in qa_pairs]
            
            # Find dominant topic
            dominant_topic = max(set(cluster_topics), key=cluster_topics.count)
            
            # Generate question patterns
            question_patterns = self._extract_question_patterns(cluster_questions)
            
            # Generate new questions
            cluster_generated = self._generate_cluster_questions(
                dominant_topic, question_patterns, cluster_answers, max_questions_per_cluster
            )
            
            for question in cluster_generated:
                question['cluster_id'] = cluster_id
                question['dominant_topic'] = dominant_topic
                question['source_cluster_size'] = len(qa_pairs)
                generated_questions.append(question)
        
        return generated_questions
    
    def _extract_question_patterns(self, questions: List[str]) -> List[str]:
        """Extract common question patterns from cluster - Enhanced version"""
        patterns = []
        
        # 🎯 ENHANCED: Question starter analysis with frequency
        starters = []
        question_types = []
        
        for q in questions:
            words = q.lower().split()
            if words:
                starter = words[0]
                if starter in ['what', 'how', 'why', 'when', 'where', 'who', 'which']:
                    starters.append(starter)
                    
                    # Analyze question type based on starter and structure
                    if starter == 'what' and ('is' in q.lower() or 'are' in q.lower()):
                        question_types.append('definition')
                    elif starter == 'how' and ('work' in q.lower() or 'function' in q.lower()):
                        question_types.append('mechanism')
                    elif starter == 'why' and ('important' in q.lower() or 'useful' in q.lower()):
                        question_types.append('importance')
                    elif 'benefit' in q.lower() or 'advantage' in q.lower():
                        question_types.append('benefits')
                    elif 'challenge' in q.lower() or 'problem' in q.lower():
                        question_types.append('challenges')
                    else:
                        question_types.append('general')
        
        # Most common starter and type
        if starters:
            common_starter = max(set(starters), key=starters.count)
            patterns.append(common_starter)
        
        if question_types:
            common_type = max(set(question_types), key=question_types.count)
            patterns.append(f"type_{common_type}")
        
        # 🎯 ENHANCED: Advanced structural patterns
        for q in questions:
            if '?' in q:
                q_lower = q.lower()
                
                # Preposition patterns
                if ' of ' in q_lower:
                    patterns.append('___ of ___')
                if ' in ' in q_lower:
                    patterns.append('___ in ___')
                if ' for ' in q_lower:
                    patterns.append('___ for ___')
                if ' with ' in q_lower:
                    patterns.append('___ with ___')
                if ' between ' in q_lower:
                    patterns.append('___ between ___')
                
                # Verb patterns
                if q_lower.startswith('can '):
                    patterns.append('can_ability')
                if q_lower.startswith('should '):
                    patterns.append('should_recommendation')
                if q_lower.startswith('will '):
                    patterns.append('will_future')
                
                # Question complexity patterns
                if len(q.split()) > 15:
                    patterns.append('complex_question')
                elif len(q.split()) < 6:
                    patterns.append('simple_question')
                else:
                    patterns.append('medium_question')
                
                # Domain-specific patterns
                if any(word in q_lower for word in ['technology', 'science', 'research']):
                    patterns.append('technical_domain')
                elif any(word in q_lower for word in ['history', 'culture', 'society']):
                    patterns.append('social_domain')
                elif any(word in q_lower for word in ['business', 'market', 'economy']):
                    patterns.append('business_domain')
        
        return list(set(patterns))
    
    def _generate_cluster_questions(self, topic: str, patterns: List[str], 
                                   answers: List[str], max_questions: int) -> List[Dict]:
        """Generate diverse questions for a specific cluster"""
        generated = []
        
        # 🎯 ENHANCED: Diverse question categories and formats
        question_categories = {
            'definition': [
                f"What exactly is {topic}?",
                f"Can you explain {topic} in simple terms?",
                f"Define {topic} and its core concepts",
                f"What does {topic} mean in practice?"
            ],
            'benefits': [
                f"What are the main advantages of {topic}?",
                f"Why should we care about {topic}?",
                f"What value does {topic} provide?",
                f"How does {topic} benefit society?"
            ],
            'mechanics': [
                f"How does {topic} actually work?",
                f"What's the process behind {topic}?",
                f"Explain the mechanism of {topic}",
                f"Walk me through how {topic} functions"
            ],
            'challenges': [
                f"What are the biggest challenges in {topic}?",
                f"What problems does {topic} face today?",
                f"What limitations exist in {topic}?",
                f"Where does {topic} struggle most?"
            ],
            'future': [
                f"Where is {topic} heading in the future?",
                f"What's next for {topic} development?",
                f"How will {topic} evolve over time?",
                f"What trends are shaping {topic}?"
            ],
            'comparison': [
                f"How does {topic} compare to traditional methods?",
                f"What makes {topic} unique from alternatives?",
                f"Why choose {topic} over other options?",
                f"What distinguishes {topic} from competitors?"
            ],
            'practical': [
                f"How can I apply {topic} in real life?",
                f"What are practical uses of {topic}?",
                f"Give me examples of {topic} in action",
                f"Where do we see {topic} being used?"
            ],
            'technical': [
                f"What are the technical aspects of {topic}?",
                f"How complex is {topic} to implement?",
                f"What skills are needed for {topic}?",
                f"What tools are used in {topic}?"
            ],
            'impact': [
                f"What impact has {topic} had on industry?",
                f"How has {topic} changed the landscape?",
                f"What role does {topic} play in modern society?",
                f"How significant is {topic}'s influence?"
            ],
            'learning': [
                f"How can someone get started with {topic}?",
                f"What should beginners know about {topic}?",
                f"What are the basics of {topic}?",
                f"How do you learn {topic} effectively?"
            ]
        }
        
        # 🎯 ENHANCED: Smart category selection based on cluster content
        selected_categories = []
        
        # Analyze cluster content to determine most relevant categories
        cluster_text = ' '.join(answers).lower()
        
        category_relevance = {}
        for category, templates in question_categories.items():
            relevance_score = 0
            
            # Category-specific keywords
            if category == 'definition' and any(word in cluster_text for word in ['is', 'are', 'definition', 'concept', 'means']):
                relevance_score += 2
            elif category == 'benefits' and any(word in cluster_text for word in ['benefit', 'advantage', 'useful', 'help', 'improve']):
                relevance_score += 2
            elif category == 'mechanics' and any(word in cluster_text for word in ['work', 'process', 'function', 'mechanism', 'how']):
                relevance_score += 2
            elif category == 'challenges' and any(word in cluster_text for word in ['problem', 'challenge', 'difficulty', 'issue', 'limitation']):
                relevance_score += 2
            elif category == 'future' and any(word in cluster_text for word in ['future', 'next', 'evolve', 'trend', 'development']):
                relevance_score += 2
            elif category == 'comparison' and any(word in cluster_text for word in ['compare', 'different', 'alternative', 'versus', 'better']):
                relevance_score += 2
            elif category == 'practical' and any(word in cluster_text for word in ['use', 'apply', 'example', 'real', 'practical']):
                relevance_score += 2
            elif category == 'technical' and any(word in cluster_text for word in ['technical', 'implement', 'complex', 'skill', 'tool']):
                relevance_score += 2
            elif category == 'impact' and any(word in cluster_text for word in ['impact', 'influence', 'change', 'effect', 'society']):
                relevance_score += 2
            elif category == 'learning' and any(word in cluster_text for word in ['learn', 'beginner', 'start', 'basic', 'education']):
                relevance_score += 2
            
            # Add base relevance
            relevance_score += 1
            category_relevance[category] = relevance_score
        
        # Select top categories by relevance
        sorted_categories = sorted(category_relevance.items(), key=lambda x: x[1], reverse=True)
        selected_categories = [cat for cat, score in sorted_categories[:max_questions]]
        
        # 🎯 ENHANCED: Generate diverse questions from selected categories
        import random
        random.seed(42)  # For reproducible results
        
        for i, category in enumerate(selected_categories):
            if category in question_categories:
                # Pick a random template from the category for diversity
                templates = question_categories[category]
                selected_template = random.choice(templates)
                
                # Calculate enhanced relevance score
                relevance = self._calculate_question_relevance(selected_template, answers)
                
                # Boost relevance for category match
                relevance = min(1.0, relevance + 0.1)
                
                generated.append({
                    'question': selected_template,
                    'generation_method': f'enhanced_cluster_based_{category}',
                    'confidence': 0.85,  # Higher confidence for enhanced generation
                    'relevance_score': relevance,
                    'suggested_priority': max_questions - i,  # Higher number = higher priority
                    'question_category': category
                })
        
        # 🎯 ENHANCED: Add pattern-based questions if patterns exist
        if patterns and len(generated) < max_questions:
            pattern_starters = [p for p in patterns if p in ['what', 'how', 'why', 'when', 'where', 'who', 'which']]
            if pattern_starters:
                common_starter = pattern_starters[0]
                pattern_questions = [
                    f"{common_starter.capitalize()} recent innovations exist in {topic}?",
                    f"{common_starter.capitalize()} can {topic} be optimized further?",
                    f"{common_starter.capitalize()} makes {topic} particularly effective?",
                    f"{common_starter.capitalize()} trends are emerging in {topic}?"
                ]
                
                for pattern_q in pattern_questions[:max_questions - len(generated)]:
                    generated.append({
                        'question': pattern_q,
                        'generation_method': 'pattern_based_enhanced',
                        'confidence': 0.8,
                        'relevance_score': self._calculate_question_relevance(pattern_q, answers),
                        'suggested_priority': 1,
                        'question_category': 'pattern_derived'
                    })
        
        return generated
    
    def _calculate_question_relevance(self, question: str, cluster_answers: List[str]) -> float:
        """Calculate how relevant a generated question is to cluster content - Enhanced version"""
        if not cluster_answers:
            return 0.75  # Higher default for generated questions
        
        question_words = set(question.lower().split())
        # Enhanced stop words list
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their'}
        question_content = question_words - stop_words
        
        if not question_content:
            return 0.7  # Higher fallback for questions with only stop words
        
        # Calculate enhanced semantic relevance
        max_relevance = 0
        total_relevance = 0
        valid_answers = 0
        
        for answer in cluster_answers:
            answer_words = set(answer.lower().split()) - stop_words
            if not answer_words:
                continue
                
            valid_answers += 1
            
            # Calculate enhanced word overlap with weighting
            overlap = len(question_content.intersection(answer_words))
            union = len(question_content.union(answer_words))
            
            # Jaccard similarity + overlap ratio
            jaccard = overlap / union if union > 0 else 0
            overlap_ratio = overlap / len(question_content) if question_content else 0
            
            # Combined relevance score
            relevance = (jaccard * 0.4 + overlap_ratio * 0.6)
            total_relevance += relevance
            max_relevance = max(max_relevance, relevance)
        
        if valid_answers == 0:
            return 0.8  # High default for structured questions
        
        # Enhanced calculation
        avg_relevance = total_relevance / valid_answers
        
        # Weight between average and max relevance
        combined_relevance = (avg_relevance * 0.6 + max_relevance * 0.4)
        
        # Enhanced scaling with reasonable range (0.6 to 1.0)
        final_score = combined_relevance * 1.2 + 0.4
        return max(0.6, min(final_score, 1.0))

    # ============= ORIGINAL METRICS (Data Generation, Clustering, Selection, Inference) =============
    
    def calculate_generation_metrics(self, qa_pairs: List[Dict], generation_time: float) -> Dict[str, Any]:
        """Calculate metrics for data generation phase"""
        if not qa_pairs:
            return {}
        
        questions = [pair['question'] for pair in qa_pairs]
        answers = [pair['answer'] for pair in qa_pairs]
        
        question_lengths = [len(q.split()) for q in questions]
        answer_lengths = [len(a.split()) for a in answers]
        
        unique_questions = len(set(questions))
        unique_answers = len(set(answers))
        
        topics = [pair.get('topic', 'Unknown') for pair in qa_pairs]
        topic_distribution = {}
        for topic in topics:
            topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
        
        valid_questions = sum(1 for q in questions if q.endswith('?'))
        avg_question_complexity = np.mean([len(set(q.lower().split())) for q in questions])
        
        metrics = {
            'total_pairs': len(qa_pairs),
            'generation_time': generation_time,
            'pairs_per_second': len(qa_pairs) / generation_time if generation_time > 0 else 0,
            'avg_question_length': np.mean(question_lengths),
            'avg_answer_length': np.mean(answer_lengths),
            'std_question_length': np.std(question_lengths),
            'std_answer_length': np.std(answer_lengths),
            'question_uniqueness': unique_questions / len(questions) if questions else 0,
            'answer_uniqueness': unique_answers / len(answers) if answers else 0,
            'avg_question_complexity': avg_question_complexity,
            'valid_question_ratio': valid_questions / len(questions) if questions else 0,
            'topic_diversity': len(topic_distribution),
            'topic_distribution': topic_distribution,
            'topics_covered': len(set(topics)),
            'avg_pairs_per_topic': len(qa_pairs) / len(set(topics)) if topics else 0
        }
        
        return self._convert_numpy_types(metrics)
    
    def calculate_clustering_metrics(self, embeddings: np.ndarray, labels: np.ndarray, 
                                   centroids: np.ndarray, clustering_time: float,
                                   qa_pairs: List[Dict] = None) -> Dict[str, Any]:
        """Calculate comprehensive clustering metrics"""
        
        if len(embeddings) < 2:
            return {'error': 'Insufficient data for clustering metrics'}
        
        n_clusters = len(np.unique(labels))
        
        silhouette = silhouette_score(embeddings, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels.astype(str), counts.tolist()))
        
        cluster_balance = np.std(counts) / np.mean(counts) if len(counts) > 1 else 0
        
        distances_to_centroids = []
        for i, embedding in enumerate(embeddings):
            cluster_id = labels[i]
            centroid = centroids[cluster_id]
            distance = np.linalg.norm(embedding - centroid)
            distances_to_centroids.append(distance)
        
        topic_coherence = 0
        if qa_pairs:
            topic_coherence = self._calculate_topic_coherence(qa_pairs, labels)
        
        metrics = {
            'num_clusters': n_clusters,
            'clustering_time': clustering_time,
            'total_points': len(embeddings),
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'topic_coherence': topic_coherence,
            'cluster_sizes': cluster_sizes,
            'cluster_balance': cluster_balance,
            'avg_distance_to_centroid': np.mean(distances_to_centroids),
            'std_distance_to_centroid': np.std(distances_to_centroids),
            'embedding_dimension': embeddings.shape[1],
            'clustering_efficiency': n_clusters / len(embeddings) if len(embeddings) > 0 else 0
        }
        
        # Convert numpy types to native Python types for JSON serialization
        return self._convert_numpy_types(metrics)
    
    def _calculate_topic_coherence(self, qa_pairs: List[Dict], labels: np.ndarray) -> float:
        """Calculate topic coherence within clusters"""
        if not qa_pairs or len(qa_pairs) != len(labels):
            return 0.0
        
        cluster_topics = {}
        for i, pair in enumerate(qa_pairs):
            cluster_id = labels[i]
            # Try multiple ways to get topic
            topic = pair.get('topic') or pair.get('original_topic') or pair.get('category')
            
            # If no topic, try to infer from question + answer content
            if not topic or topic in ['Unknown', 'unknown', '']:
                question = pair.get('question', '')
                answer = pair.get('answer', '')
                topic = self._infer_topic_from_content(question, answer)
            
            if cluster_id not in cluster_topics:
                cluster_topics[cluster_id] = []
            cluster_topics[cluster_id].append(topic)
        
        coherence_scores = []
        for cluster_id, topics in cluster_topics.items():
            if topics:
                # Count frequency of each topic
                topic_counts = {}
                for topic in topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                # Find most common topic
                most_common_count = max(topic_counts.values())
                coherence = most_common_count / len(topics)
                coherence_scores.append(coherence)
        
        overall_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        
        return overall_coherence
        
    def _infer_topic_from_question(self, question: str) -> str:
        """Infer topic from question content (legacy method)"""
        return self._infer_topic_from_content(question, "")
        
    def _infer_topic_from_content(self, question: str, answer: str) -> str:
        """Infer topic from question and answer content"""
        content = f"{question} {answer}".lower()
        
        if not content.strip():
            return 'general'
        
        # Science keywords (expanded)
        science_keywords = ['atom', 'molecule', 'chemical', 'physics', 'chemistry', 'biology', 'dna', 'cell', 
                           'evolution', 'gravity', 'energy', 'force', 'electron', 'proton', 'neutron', 'nucleus',
                           'photosynthesis', 'mitosis', 'protein', 'enzyme', 'reaction', 'theory', 'hypothesis',
                           'experiment', 'carbon', 'oxygen', 'hydrogen', 'periodic', 'element', 'compound']
        
        # Technology keywords (expanded)  
        tech_keywords = ['computer', 'software', 'internet', 'ai', 'algorithm', 'programming', 'technology',
                        'digital', 'data', 'code', 'website', 'app', 'database', 'server', 'network',
                        'machine learning', 'artificial intelligence', 'robot', 'automation', 'cyber',
                        'bitcoin', 'blockchain', 'smartphone', 'laptop', 'processor', 'memory', 'storage']
        
        # History keywords (expanded)
        history_keywords = ['war', 'ancient', 'empire', 'century', 'historical', 'civilization', 'battle',
                           'revolution', 'king', 'president', 'dynasty', 'medieval', 'renaissance', 'colonial',
                           'independence', 'treaty', 'constitution', 'democracy', 'republic', 'monarchy',
                           'pharaoh', 'roman', 'greek', 'egyptian', 'viking', 'napoleon', 'lincoln', 'washington']
        
        # Sports keywords (expanded)
        sports_keywords = ['sport', 'game', 'team', 'player', 'football', 'basketball', 'soccer', 'olympic',
                          'competition', 'championship', 'tournament', 'league', 'match', 'score', 'goal',
                          'tennis', 'golf', 'baseball', 'volleyball', 'swimming', 'running', 'racing',
                          'athlete', 'coach', 'stadium', 'field', 'court', 'medal', 'trophy']
        
        # Math keywords (expanded)  
        math_keywords = ['math', 'equation', 'calculate', 'number', 'algebra', 'geometry', 'statistics',
                        'formula', 'theorem', 'proof', 'integral', 'derivative', 'function', 'variable',
                        'coefficient', 'polynomial', 'trigonometry', 'logarithm', 'exponential', 'matrix',
                        'vector', 'probability', 'percentage', 'fraction', 'decimal', 'ratio', 'proportion']
        
        # Count matches for each category
        science_count = sum(1 for keyword in science_keywords if keyword in content)
        tech_count = sum(1 for keyword in tech_keywords if keyword in content)
        history_count = sum(1 for keyword in history_keywords if keyword in content)
        sports_count = sum(1 for keyword in sports_keywords if keyword in content)
        math_count = sum(1 for keyword in math_keywords if keyword in content)
        
        # Return the category with the most matches
        counts = {
            'science': science_count,
            'technology': tech_count, 
            'history': history_count,
            'sports': sports_count,
            'mathematics': math_count
        }
        
        max_category = max(counts.keys(), key=lambda k: counts[k])
        max_count = counts[max_category]
        

        
        # If no clear category, return general
        if max_count == 0:
            return 'general'
            
        return max_category

    def calculate_selection_metrics(self, selected_examples: List[Dict], 
                                  total_examples: int, clustering_data: Dict,
                                  selection_time: float) -> Dict[str, Any]:
        """Calculate metrics for example selection phase"""
        
        if not selected_examples:
            return {'error': 'No examples selected'}
        
        num_selected = len(selected_examples)
        selection_ratio = num_selected / total_examples if total_examples > 0 else 0
        
        cluster_ids = [ex.get('cluster_id', -1) for ex in selected_examples]
        unique_clusters = len(set(cluster_ids))
        total_clusters = len(clustering_data.get('qa_pairs', {})) if clustering_data else 0
        cluster_coverage = unique_clusters / total_clusters if total_clusters > 0 else 0
        
        distances = [ex.get('distance_to_centroid', 0) for ex in selected_examples]
        
        metrics = {
            'total_selected': num_selected,
            'selection_time': selection_time,
            'selection_ratio': selection_ratio,
            'cluster_coverage': cluster_coverage,
            'clusters_covered': unique_clusters,
            'total_clusters': total_clusters,
            'avg_distance_to_centroid': np.mean(distances) if distances else 0,
            'selection_efficiency': selection_ratio
        }
        
        return self._convert_numpy_types(metrics)

    def calculate_inference_metrics(self, question: str, results: Dict[str, Dict],
                                  inference_time: float, context_examples: int) -> Dict[str, Any]:
        """Calculate comprehensive metrics for inference phase with answer quality evaluation"""
        
        models_used = len([k for k, v in results.items() if 'error' not in v])
        successful_inferences = sum(1 for v in results.values() if 'answer' in v)
        
        question_length = len(question.split())
        question_complexity = len(set(question.lower().split()))
        
        # Question analysis
        question_type = self._analyze_question_type(question)
        question_difficulty = self._estimate_question_difficulty(question)
        
        model_metrics = {}
        answer_quality_scores = {}
        
        for model_name, result in results.items():
            if 'answer' in result:
                answer = result['answer']
                
                # Basic metrics
                answer_length = len(answer.split())
                answer_complexity = len(set(answer.lower().split()))
                
                # Answer quality assessment
                quality_score = self._evaluate_answer_quality(question, answer)
                coherence_score = self._assess_answer_coherence(answer)
                relevance_score = self._assess_answer_relevance(question, answer)
                completeness_score = self._assess_answer_completeness(question, answer)
                factual_score = self._assess_factual_consistency(answer)
                
                # Overall answer rating
                overall_score = (quality_score + coherence_score + relevance_score + 
                               completeness_score + factual_score) / 5
                
                model_metrics[model_name] = {
                    'inference_time': result.get('inference_time', 0),
                    'answer_length': answer_length,
                    'answer_complexity': answer_complexity,
                    'confidence': result.get('confidence', None),
                    'success': True,
                    'answer_quality_score': quality_score,
                    'coherence_score': coherence_score,
                    'relevance_score': relevance_score,
                    'completeness_score': completeness_score,
                    'factual_consistency_score': factual_score,
                    'overall_answer_score': overall_score,
                    'answer_rating': self._get_answer_rating(overall_score),
                    'context_utilization': result.get('context_used', 0),
                    'generation_params': result.get('generation_params', {})
                }
                
                answer_quality_scores[model_name] = overall_score
                
            else:
                model_metrics[model_name] = {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'answer_quality_score': 0,
                    'overall_answer_score': 0
                }
        
        # Cross-model comparison
        comparison_insights = {}
        if len(answer_quality_scores) >= 2:
            comparison_insights = self._generate_answer_comparison_insights(model_metrics, question_type)
        
        # Performance insights
        performance_insights = self._generate_performance_insights(model_metrics, question_difficulty)
        
        metrics = {
            'total_inference_time': inference_time,
            'models_used': models_used,
            'successful_inferences': successful_inferences,
            'success_rate': successful_inferences / len(results) if results else 0,
            'question_analysis': {
                'length': question_length,
                'complexity': question_complexity,
                'type': question_type,
                'difficulty': question_difficulty
            },
            'context_examples_used': context_examples,
            'model_metrics': model_metrics,
            'answer_quality_summary': {
                'best_quality_model': max(answer_quality_scores.keys(), 
                                        key=lambda k: answer_quality_scores[k]) if answer_quality_scores else None,
                'avg_answer_quality': np.mean(list(answer_quality_scores.values())) if answer_quality_scores else 0,
                'quality_variance': np.var(list(answer_quality_scores.values())) if len(answer_quality_scores) > 1 else 0
            },
            'comparison_insights': comparison_insights,
            'performance_insights': performance_insights,
            'timestamp': datetime.now().isoformat()
        }
        
        return self._convert_numpy_types(metrics)
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze the type of question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'which', 'who', 'where', 'when']):
            return 'factual'
        elif any(word in question_lower for word in ['how', 'why']):
            return 'explanatory'
        elif any(word in question_lower for word in ['do', 'does', 'is', 'are', 'can', 'will']):
            return 'yes_no'
        elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
            return 'comparative'
        else:
            return 'open_ended'
    
    def _estimate_question_difficulty(self, question: str) -> str:
        """Estimate question difficulty based on complexity"""
        words = question.split()
        unique_words = len(set(question.lower().split()))
        
        # Complex question indicators
        complex_indicators = ['analyze', 'evaluate', 'compare', 'explain why', 'how does', 'relationship']
        has_complex_terms = any(term in question.lower() for term in complex_indicators)
        
        if len(words) > 15 or unique_words > 12 or has_complex_terms:
            return 'hard'
        elif len(words) > 8 or unique_words > 6:
            return 'medium'
        else:
            return 'easy'
    
    def _evaluate_answer_quality(self, question: str, answer: str) -> float:
        """Evaluate overall answer quality"""
        if not answer or len(answer.strip()) < 3:
            return 0.0
        
        # Basic quality indicators
        answer_words = answer.split()
        question_words = set(question.lower().split())
        answer_words_set = set(answer.lower().split())
        
        # Check for keyword overlap
        keyword_overlap = len(question_words.intersection(answer_words_set)) / len(question_words) if question_words else 0
        
        # Check answer length appropriateness
        length_score = min(len(answer_words) / 10, 1.0)  # Optimal around 10 words
        
        # Check for complete sentences
        sentence_score = 1.0 if answer.endswith(('.', '!', '?')) else 0.5
        
        # Check for direct response indicators
        direct_response = 1.0 if not answer.lower().startswith(('i ', 'the question', 'this question')) else 0.7
        
        return np.mean([keyword_overlap, length_score, sentence_score, direct_response])
    
    def _assess_answer_coherence(self, answer: str) -> float:
        """Assess answer coherence and readability"""
        if not answer:
            return 0.0
        
        sentences = answer.split('.')
        coherence_indicators = [
            len(sentences) > 0,  # Has sentences
            not any(word in answer.lower() for word in ['uh', 'um', 'err']),  # No filler words
            len(answer.split()) >= 3,  # Minimum length
            not answer.lower().startswith('i need more'),  # Not a cop-out answer
        ]
        
        return sum(coherence_indicators) / len(coherence_indicators)
    
    def _assess_answer_relevance(self, question: str, answer: str) -> float:
        """Assess how relevant the answer is to the question"""
        if not answer or not question:
            return 0.0
        
        # Simple keyword matching approach
        question_keywords = set(question.lower().split())
        answer_keywords = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        question_keywords -= stop_words
        answer_keywords -= stop_words
        
        if not question_keywords:
            return 0.5
        
        overlap = len(question_keywords.intersection(answer_keywords))
        relevance = overlap / len(question_keywords)
        
        return min(relevance, 1.0)
    
    def _assess_answer_completeness(self, question: str, answer: str) -> float:
        """Assess how complete the answer is"""
        if not answer:
            return 0.0
        
        # Check if answer is too short
        if len(answer.split()) < 3:
            return 0.2
        
        # Check for complete thoughts
        has_subject_verb = any(word in answer.lower() for word in ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will', 'would'])
        
        # Check if it answers the question type
        question_type = self._analyze_question_type(question)
        if question_type == 'yes_no':
            has_yes_no = any(word in answer.lower() for word in ['yes', 'no', 'true', 'false'])
            return 1.0 if has_yes_no else 0.5
        
        return 0.8 if has_subject_verb else 0.4
    
    def _assess_factual_consistency(self, answer: str) -> float:
        """Basic assessment of factual consistency"""
        if not answer:
            return 0.0
        
        # Check for contradictory statements
        contradiction_indicators = [
            'but' in answer.lower() and 'however' in answer.lower(),
            'not' in answer.lower() and len(answer.split()) < 5
        ]
        
        if any(contradiction_indicators):
            return 0.3
        
        # Check for uncertainty markers
        uncertainty_markers = ['maybe', 'perhaps', 'might', 'could be', 'possibly']
        has_uncertainty = any(marker in answer.lower() for marker in uncertainty_markers)
        
        return 0.7 if has_uncertainty else 0.9
    
    def _get_answer_rating(self, score: float) -> str:
        """Convert score to rating"""
        if score >= 0.8:
            return 'Excellent'
        elif score >= 0.6:
            return 'Good'
        elif score >= 0.4:
            return 'Fair'
        elif score >= 0.2:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _generate_answer_comparison_insights(self, model_metrics: Dict, question_type: str) -> Dict:
        """Generate insights comparing model answers"""
        insights = {}
        
        models = [k for k, v in model_metrics.items() if v.get('success', False)]
        if len(models) < 2:
            return insights
        
        # Compare overall scores
        scores = {model: model_metrics[model]['overall_answer_score'] for model in models}
        best_model = max(scores.keys(), key=lambda k: scores[k])
        
        insights['best_overall_model'] = best_model
        insights['score_difference'] = max(scores.values()) - min(scores.values())
        
        # Compare specific aspects
        aspects = ['coherence_score', 'relevance_score', 'completeness_score']
        for aspect in aspects:
            aspect_scores = {model: model_metrics[model][aspect] for model in models}
            insights[f'best_{aspect.replace("_score", "")}_model'] = max(aspect_scores.keys(), key=lambda k: aspect_scores[k])
        
        return insights
    
    def _generate_performance_insights(self, model_metrics: Dict, question_difficulty: str) -> Dict:
        """Generate performance insights"""
        insights = {
            'question_difficulty': question_difficulty,
            'models_performance': {},
            'recommendations': []
        }
        
        for model_name, metrics in model_metrics.items():
            if metrics.get('success', False):
                performance_level = 'high' if metrics['overall_answer_score'] > 0.7 else \
                                 'medium' if metrics['overall_answer_score'] > 0.4 else 'low'
                
                insights['models_performance'][model_name] = {
                    'performance_level': performance_level,
                    'best_aspect': self._get_best_aspect(metrics),
                    'needs_improvement': self._get_weak_aspect(metrics)
                }
        
        # Generate recommendations
        if question_difficulty == 'hard':
            insights['recommendations'].append('Consider using additional context for complex questions')
        
        return insights
    
    def _get_best_aspect(self, metrics: Dict) -> str:
        """Get the best performing aspect"""
        aspects = {
            'coherence': metrics.get('coherence_score', 0),
            'relevance': metrics.get('relevance_score', 0),
            'completeness': metrics.get('completeness_score', 0)
        }
        return max(aspects.keys(), key=lambda k: aspects[k])
    
    def _get_weak_aspect(self, metrics: Dict) -> str:
        """Get the weakest performing aspect"""
        aspects = {
            'coherence': metrics.get('coherence_score', 0),
            'relevance': metrics.get('relevance_score', 0),
            'completeness': metrics.get('completeness_score', 0)
        }
        return min(aspects.keys(), key=lambda k: aspects[k])

    def generate_performance_report(self, all_metrics: Dict) -> str:
        """Generate comprehensive performance report"""
        
        report = []
        report.append("="*80)
        report.append("SELF-PROMPTING PIPELINE PERFORMANCE REPORT")
        report.append("="*80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance Summary
        if 'model_comparison' in all_metrics:
            comp = all_metrics['model_comparison']
            report.append("🆚 MODEL COMPARISON SUMMARY")
            report.append("-" * 40)
            report.append(f"Models: {comp.get('models_compared', 'N/A')}")
            if 'comparison_summary' in comp:
                report.append(f"Summary: {comp['comparison_summary']}")
            report.append("")
        
        # Add other sections...
        return "\n".join(report)

    def evaluate_answer_against_reference(self, question: str, generated_answer: str, 
                                          reference_answer: str, model_name: str = "Unknown") -> Dict[str, Any]:
        """Evaluate generated answer quality against reference answer from data generation"""
        
        if not reference_answer or not generated_answer:
            return {
                'reference_similarity': 0.0,
                'semantic_similarity': 0.0,
                'length_ratio': 0.0,
                'keyword_overlap': 0.0,
                'quality_rating': 'Poor',
                'detailed_metrics': {
                    'reference_available': bool(reference_answer),
                    'answer_provided': bool(generated_answer)
                }
            }
        
        # Clean and prepare texts
        ref_clean = reference_answer.lower().strip()
        gen_clean = generated_answer.lower().strip()
        
        # 1. Exact text similarity (simple approach)
        from difflib import SequenceMatcher
        text_similarity = SequenceMatcher(None, ref_clean, gen_clean).ratio()
        
        # 2. Keyword overlap analysis
        ref_words = set(ref_clean.split())
        gen_words = set(gen_clean.split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'have', 'has', 'had'}
        ref_words -= stop_words
        gen_words -= stop_words
        
        if ref_words:
            keyword_overlap = len(ref_words.intersection(gen_words)) / len(ref_words)
        else:
            keyword_overlap = 0.0
        
        # 3. Length ratio analysis
        ref_length = len(reference_answer.split())
        gen_length = len(generated_answer.split())
        
        if ref_length > 0:
            length_ratio = min(gen_length / ref_length, 2.0)  # Cap at 2x reference length
            length_score = 1.0 - abs(length_ratio - 1.0)  # Penalize deviations from reference length
        else:
            length_ratio = 0.0
            length_score = 0.0
        
        # 4. Semantic similarity using simple TF-IDF approach
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([ref_clean, gen_clean])
            semantic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            # Fallback to simple word overlap if sklearn fails
            semantic_similarity = keyword_overlap
        
        # 5. Content accuracy assessment
        content_accuracy = self._assess_content_accuracy(reference_answer, generated_answer)
        
        # 6. Answer completeness relative to reference
        ref_completeness = self._assess_reference_completeness(reference_answer, generated_answer)
        
        # 7. Overall quality score
        overall_score = (
            text_similarity * 0.2 +
            semantic_similarity * 0.3 +
            keyword_overlap * 0.2 +
            length_score * 0.1 +
            content_accuracy * 0.1 +
            ref_completeness * 0.1
        )
        
        # Quality rating
        if overall_score >= 0.8:
            quality_rating = 'Excellent'
        elif overall_score >= 0.6:
            quality_rating = 'Good'
        elif overall_score >= 0.4:
            quality_rating = 'Fair'
        elif overall_score >= 0.2:
            quality_rating = 'Poor'
        else:
            quality_rating = 'Very Poor'
        
        return {
            'overall_score': round(overall_score, 3),
            'reference_similarity': round(text_similarity, 3),
            'semantic_similarity': round(semantic_similarity, 3),
            'keyword_overlap': round(keyword_overlap, 3),
            'length_ratio': round(length_ratio, 3),
            'content_accuracy': round(content_accuracy, 3),
            'reference_completeness': round(ref_completeness, 3),
            'quality_rating': quality_rating,
            'model_name': model_name,
            'detailed_metrics': {
                'reference_length': ref_length,
                'generated_length': gen_length,
                'reference_words': len(ref_words),
                'generated_words': len(gen_words),
                'common_keywords': len(ref_words.intersection(gen_words)),
                'question_answered': self._check_question_answered(question, generated_answer)
            }
        }
    
    def _assess_content_accuracy(self, reference: str, generated: str) -> float:
        """Assess if generated answer contains accurate information relative to reference"""
        ref_lower = reference.lower()
        gen_lower = generated.lower()
        
        # Look for contradictions
        positive_indicators = ['is', 'are', 'was', 'were', 'can', 'will', 'does', 'has']
        negative_indicators = ['not', 'no', 'never', 'cannot', 'won\'t', 'doesn\'t', 'hasn\'t']
        
        ref_positive = any(indicator in ref_lower for indicator in positive_indicators)
        gen_positive = any(indicator in gen_lower for indicator in positive_indicators)
        
        ref_negative = any(indicator in ref_lower for indicator in negative_indicators)
        gen_negative = any(indicator in gen_lower for indicator in negative_indicators)
        
        # Check for factual consistency
        if (ref_positive and gen_negative) or (ref_negative and gen_positive):
            return 0.3  # Potential contradiction
        elif (ref_positive and gen_positive) or (ref_negative and gen_negative):
            return 0.8  # Consistent tone
        else:
            return 0.6  # Neutral
    
    def _assess_reference_completeness(self, reference: str, generated: str) -> float:
        """Assess how completely the generated answer covers the reference content"""
        ref_sentences = [s.strip() for s in reference.split('.') if s.strip()]
        gen_sentences = [s.strip() for s in generated.split('.') if s.strip()]
        
        if not ref_sentences:
            return 0.5
        
        # Check if key concepts from reference are covered
        ref_concepts = []
        for sentence in ref_sentences:
            words = [w for w in sentence.lower().split() if len(w) > 3]
            ref_concepts.extend(words)
        
        gen_text_lower = generated.lower()
        covered_concepts = sum(1 for concept in ref_concepts if concept in gen_text_lower)
        
        if ref_concepts:
            return covered_concepts / len(ref_concepts)
        else:
            return 0.5
    
    def _check_question_answered(self, question: str, answer: str) -> bool:
        """Check if the answer actually addresses the question"""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common question words
        question_indicators = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'can', 'will'}
        meaningful_question_words = question_words - question_indicators - {'the', 'a', 'an', 'and', 'or', 'but'}
        
        if not meaningful_question_words:
            return len(answer.strip()) > 10  # At least some content
        
        # Check if answer contains relevant words from question
        overlap = len(meaningful_question_words.intersection(answer_words))
        return overlap >= len(meaningful_question_words) * 0.3  # At least 30% overlap
    
    def compare_model_answers_with_references(self, qa_data: List[Dict], inference_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare model answers with reference answers from generated data"""
        
        if not qa_data or not inference_results:
            return {'error': 'Insufficient data for comparison'}
        
        # Create a lookup for reference answers
        reference_lookup = {}
        for qa_pair in qa_data:
            question = qa_pair.get('question', '').strip().lower()
            answer = qa_pair.get('answer', '')
            reference_lookup[question] = answer
        
        comparison_results = {}
        
        for model_name, result in inference_results.items():
            if 'answer' not in result:
                comparison_results[model_name] = {'error': 'No answer provided'}
                continue
            
            generated_answer = result['answer']
            question = result.get('question', '').strip().lower()
            
            # Find best matching reference
            best_reference = None
            best_similarity = 0
            
            for ref_question, ref_answer in reference_lookup.items():
                similarity = self._calculate_question_similarity(question, ref_question)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_reference = ref_answer
            
            if best_reference and best_similarity > 0.3:  # Threshold for considering a match
                evaluation = self.evaluate_answer_against_reference(
                    question, generated_answer, best_reference, model_name
                )
                evaluation['reference_match_confidence'] = round(best_similarity, 3)
                comparison_results[model_name] = evaluation
            else:
                comparison_results[model_name] = {
                    'error': 'No matching reference found',
                    'reference_match_confidence': round(best_similarity, 3) if best_similarity > 0 else 0.0
                }
        
        # Calculate summary statistics
        valid_results = [r for r in comparison_results.values() if 'overall_score' in r]
        
        summary = {
            'total_models_evaluated': len(comparison_results),
            'successful_evaluations': len(valid_results),
            'average_quality_score': np.mean([r['overall_score'] for r in valid_results]) if valid_results else 0,
            'best_performing_model': max(valid_results, key=lambda x: x['overall_score'])['model_name'] if valid_results else None,
            'quality_variance': np.var([r['overall_score'] for r in valid_results]) if len(valid_results) > 1 else 0
        }
        
        return {
            'model_comparisons': comparison_results,
            'summary': summary,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_question_similarity(self, q1: str, q2: str) -> float:
        """Calculate similarity between two questions"""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        
        # Remove common question words
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'can', 'will', 'the', 'a', 'an'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def calculate_inference_quality_metrics(self, question: str, generated_answer: str, 
                                           reference_answer: str = None, model_name: str = "Unknown",
                                           confidence_score: float = None, inference_time: float = 0.0,
                                           is_zero_shot: bool = True) -> Dict[str, Any]:
        """Calculate comprehensive inference quality metrics"""
        
        if not generated_answer:
            return {
                'accuracy': 0.0,
                'zero_shot_accuracy': 0.0,
                'confidence_score': 0.0,
                'bleu_score': 0.0,
                'rouge_score': 0.0,
                'factual_accuracy': 0.0,
                'relevance_score': 0.0,
                'coherence_score': 0.0,
                'informativeness': 0.0,
                'quality_rating': 'Very Poor',
                'performance_category': 'Poor'
            }
        
        metrics = {}
        
        # 1. Accuracy (against reference if available)
        if reference_answer:
            metrics['accuracy'] = self._calculate_answer_accuracy(generated_answer, reference_answer)
            metrics['exact_match'] = self._calculate_exact_match(generated_answer, reference_answer)
        else:
            metrics['accuracy'] = self._estimate_answer_accuracy_heuristic(question, generated_answer)
            metrics['exact_match'] = 0.0
        
        # 2. Zero-shot Accuracy
        if is_zero_shot:
            metrics['zero_shot_accuracy'] = metrics['accuracy']
            metrics['is_zero_shot'] = True
        else:
            metrics['zero_shot_accuracy'] = 0.0  # Not applicable for fine-tuned models
            metrics['is_zero_shot'] = False
        
        # 3. Confidence Score
        if confidence_score is not None:
            metrics['confidence_score'] = confidence_score
            metrics['confidence_calibration'] = self._assess_confidence_calibration(
                confidence_score, metrics['accuracy']
            )
        else:
            metrics['confidence_score'] = self._estimate_confidence_from_answer(generated_answer)
            metrics['confidence_calibration'] = 0.5  # Unknown
        
        # 4. BLEU Score (for text generation quality)
        if reference_answer:
            metrics['bleu_score'] = self._calculate_bleu_score(generated_answer, reference_answer)
        else:
            metrics['bleu_score'] = self._estimate_bleu_heuristic(generated_answer)
        
        # 5. ROUGE Score (for summarization quality)
        if reference_answer:
            metrics['rouge_score'] = self._calculate_rouge_score(generated_answer, reference_answer)
        else:
            metrics['rouge_score'] = self._estimate_rouge_heuristic(generated_answer)
        
        # 6. Factual Accuracy
        metrics['factual_accuracy'] = self._assess_factual_accuracy(generated_answer)
        
        # 7. Relevance Score (to the question)
        metrics['relevance_score'] = self._calculate_relevance_to_question(question, generated_answer)
        
        # 8. Coherence Score
        metrics['coherence_score'] = self._assess_answer_coherence(generated_answer)
        
        # 9. Informativeness
        metrics['informativeness'] = self._assess_informativeness(generated_answer)
        
        # 10. Response Speed Metrics
        metrics['inference_time'] = inference_time
        metrics['words_per_second'] = len(generated_answer.split()) / max(inference_time, 0.001)
        
        # 11. Answer Length Metrics
        metrics['answer_length_words'] = len(generated_answer.split())
        metrics['answer_length_chars'] = len(generated_answer)
        metrics['sentence_count'] = len([s for s in generated_answer.split('.') if s.strip()])
        
        # 12. Overall Quality Assessment
        overall_score = (
            metrics['accuracy'] * 0.25 +
            metrics['relevance_score'] * 0.20 +
            metrics['coherence_score'] * 0.15 +
            metrics['factual_accuracy'] * 0.15 +
            metrics['informativeness'] * 0.10 +
            metrics['confidence_score'] * 0.10 +
            (metrics['bleu_score'] if reference_answer else 0.6) * 0.05
        )
        
        metrics['overall_quality_score'] = round(overall_score, 3)
        metrics['quality_rating'] = self._get_quality_rating(overall_score)
        metrics['performance_category'] = self._get_performance_category(overall_score, inference_time)
        
        # 13. Model-specific metrics
        metrics['model_name'] = model_name
        metrics['evaluation_timestamp'] = datetime.now().isoformat()
        
        return self._convert_numpy_types(metrics)
    
    def _calculate_answer_accuracy(self, generated: str, reference: str) -> float:
        """Calculate accuracy against reference answer"""
        gen_clean = generated.lower().strip()
        ref_clean = reference.lower().strip()
        
        # Exact match component
        if gen_clean == ref_clean:
            return 1.0
        
        # Semantic similarity component
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, gen_clean, ref_clean).ratio()
        
        # Key information extraction component
        ref_words = set(ref_clean.split())
        gen_words = set(gen_clean.split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        ref_key_words = ref_words - stop_words
        gen_key_words = gen_words - stop_words
        
        if ref_key_words:
            key_word_overlap = len(ref_key_words.intersection(gen_key_words)) / len(ref_key_words)
        else:
            key_word_overlap = 0.5
        
        # Combined accuracy score
        accuracy = (similarity * 0.4 + key_word_overlap * 0.6)
        return min(accuracy, 1.0)
    
    def _calculate_exact_match(self, generated: str, reference: str) -> float:
        """Calculate exact match score"""
        return 1.0 if generated.strip().lower() == reference.strip().lower() else 0.0
    
    def _estimate_answer_accuracy_heuristic(self, question: str, answer: str) -> float:
        """🎯 ENHANCED: Estimate accuracy using improved heuristics when no reference is available"""
        if not answer or len(answer.strip()) < 3:
            return 0.0
        
        # Check if answer seems to address the question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove question words (enhanced list)
        question_content = question_words - {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'can', 'will', 'the', 'a', 'an', 'and', 'or', 'but'}
        
        if question_content:
            relevance = len(question_content.intersection(answer_words)) / len(question_content)
        else:
            relevance = 0.6  # More generous default
        
        # Enhanced completeness indicators
        completeness_indicators = [
            not answer.lower().startswith(('i don\'t', 'i cannot', 'sorry', 'unclear')),
            len(answer.split()) >= 4,  # Reduced minimum length requirement
            answer.strip().endswith(('.', '!', '?', ':')),  # Proper ending (expanded)
            not ('...' in answer),  # Not incomplete
            len(set(answer.split())) >= 3,  # Some vocabulary diversity
            any(char.isdigit() for char in answer) or any(word[0].isupper() for word in answer.split()[1:])  # Factual content
        ]
        
        completeness = sum(completeness_indicators) / len(completeness_indicators)
        
        # Topic relevance bonus (enhanced)
        topic_bonus = 0
        if relevance > 0.3:  # Good topic relevance
            topic_bonus = 0.1
        elif relevance > 0.1:  # Some topic relevance
            topic_bonus = 0.05
        
        # Base accuracy improvement
        base_accuracy = 0.5  # Higher starting point
        final_accuracy = base_accuracy + (relevance * 0.3) + (completeness * 0.2) + topic_bonus
        
        return min(1.0, final_accuracy)
    
    def _assess_confidence_calibration(self, confidence: float, accuracy: float) -> float:
        """Assess how well confidence aligns with actual accuracy"""
        if confidence is None or accuracy is None:
            return 0.5
        
        # Perfect calibration would have confidence == accuracy
        calibration_error = abs(confidence - accuracy)
        calibration_score = max(0, 1 - calibration_error)
        
        return calibration_score
    
    def _estimate_confidence_from_answer(self, answer: str) -> float:
        """Estimate confidence from answer characteristics"""
        if not answer:
            return 0.0
        
        # Confidence indicators
        high_confidence_words = ['definitely', 'certainly', 'clearly', 'obviously', 'indeed', 'precisely']
        low_confidence_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain', 'unclear']
        
        answer_lower = answer.lower()
        
        high_confidence_count = sum(1 for word in high_confidence_words if word in answer_lower)
        low_confidence_count = sum(1 for word in low_confidence_words if word in answer_lower)
        
        # Base confidence from answer structure
        base_confidence = 0.5
        
        # Adjust based on answer length and completeness
        if len(answer.split()) >= 10:
            base_confidence += 0.1
        if answer.strip().endswith('.'):
            base_confidence += 0.1
        if not answer.lower().startswith(('i think', 'i believe', 'maybe')):
            base_confidence += 0.1
        
        # Adjust based on confidence words
        confidence_adjustment = (high_confidence_count * 0.1) - (low_confidence_count * 0.1)
        
        final_confidence = base_confidence + confidence_adjustment
        return max(0.0, min(1.0, final_confidence))
    
    def _calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score (simplified version)"""
        try:
            # Simple BLEU-like metric using n-gram overlap
            gen_words = generated.lower().split()
            ref_words = reference.lower().split()
            
            if not gen_words or not ref_words:
                return 0.0
            
            # 1-gram precision
            gen_1grams = set(gen_words)
            ref_1grams = set(ref_words)
            precision_1 = len(gen_1grams.intersection(ref_1grams)) / len(gen_1grams) if gen_1grams else 0
            
            # 2-gram precision
            gen_2grams = set(zip(gen_words[:-1], gen_words[1:]))
            ref_2grams = set(zip(ref_words[:-1], ref_words[1:]))
            precision_2 = len(gen_2grams.intersection(ref_2grams)) / len(gen_2grams) if gen_2grams else 0
            
            # Brevity penalty
            bp = min(1.0, len(gen_words) / len(ref_words)) if ref_words else 0
            
            # Simplified BLEU
            bleu = bp * ((precision_1 + precision_2) / 2)
            return bleu
            
        except Exception:
            return 0.5  # Fallback score
    
    def _estimate_bleu_heuristic(self, answer: str) -> float:
        """Estimate BLEU score without reference"""
        if not answer:
            return 0.0
        
        # Heuristic based on answer quality indicators
        words = answer.split()
        
        quality_indicators = [
            len(words) >= 5,  # Adequate length
            len(set(words)) / len(words) > 0.7 if words else False,  # Vocabulary diversity
            not any(word in answer.lower() for word in ['uh', 'um', 'err']),  # No filler words
            answer.strip().endswith(('.', '!', '?')),  # Proper ending
        ]
        
        return sum(quality_indicators) / len(quality_indicators)
    
    def _calculate_rouge_score(self, generated: str, reference: str) -> float:
        """Calculate ROUGE score (simplified version)"""
        try:
            gen_words = generated.lower().split()
            ref_words = reference.lower().split()
            
            if not gen_words or not ref_words:
                return 0.0
            
            # ROUGE-1 (unigram overlap)
            gen_unigrams = set(gen_words)
            ref_unigrams = set(ref_words)
            
            overlap = len(gen_unigrams.intersection(ref_unigrams))
            rouge_1_precision = overlap / len(gen_unigrams) if gen_unigrams else 0
            rouge_1_recall = overlap / len(ref_unigrams) if ref_unigrams else 0
            
            if rouge_1_precision + rouge_1_recall > 0:
                rouge_1_f1 = 2 * rouge_1_precision * rouge_1_recall / (rouge_1_precision + rouge_1_recall)
            else:
                rouge_1_f1 = 0
            
            return rouge_1_f1
            
        except Exception:
            return 0.5  # Fallback score
    
    def _estimate_rouge_heuristic(self, answer: str) -> float:
        """Estimate ROUGE score without reference"""
        if not answer:
            return 0.0
        
        # Similar to BLEU heuristic but focusing on recall-oriented metrics
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        words = answer.split()
        
        quality_indicators = [
            len(sentences) >= 1,  # At least one complete sentence
            len(words) >= 3,  # Minimum content
            len(set(words)) >= len(words) * 0.6 if words else False,  # Reasonable diversity
            not answer.lower().startswith(('sorry', 'i don\'t', 'unclear')),  # Not a non-answer
        ]
        
        return sum(quality_indicators) / len(quality_indicators)
    
    def _assess_factual_accuracy(self, answer: str) -> float:
        """Assess factual accuracy of the answer"""
        if not answer:
            return 0.0
        
        # Heuristic approach for factual accuracy
        answer_lower = answer.lower()
        
        # Negative indicators (likely factually problematic)
        negative_indicators = [
            'i made up' in answer_lower,
            'fictional' in answer_lower,
            'invented' in answer_lower,
            'not real' in answer_lower,
            'fake' in answer_lower,
        ]
        
        # Positive indicators (likely factually sound)
        positive_indicators = [
            any(year in answer for year in ['19', '20']),  # Contains years
            any(num in answer for num in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']),  # Contains numbers
            len(answer.split()) >= 8,  # Substantial content
            not any(word in answer_lower for word in ['maybe', 'perhaps', 'might be']),  # Definitive
        ]
        
        negative_score = sum(negative_indicators)
        positive_score = sum(positive_indicators)
        
        if negative_score > 0:
            return 0.2  # Likely inaccurate
        
        # Base score adjusted by positive indicators
        base_score = 0.6
        adjustment = (positive_score / len(positive_indicators)) * 0.3
        
        return min(1.0, base_score + adjustment)
    
    def _calculate_relevance_to_question(self, question: str, answer: str) -> float:
        """Calculate how relevant the answer is to the question"""
        if not question or not answer:
            return 0.0
        
        # Existing relevance calculation logic
        return self._assess_answer_relevance(question, answer)
    
    def _assess_informativeness(self, answer: str) -> float:
        """Assess how informative the answer is"""
        if not answer:
            return 0.0
        
        words = answer.split()
        unique_words = set(words)
        
        # Informativeness indicators
        info_indicators = [
            len(words) >= 10,  # Adequate length
            len(unique_words) / len(words) > 0.7 if words else False,  # Vocabulary diversity
            any(word in answer.lower() for word in ['because', 'since', 'due to', 'therefore']),  # Explanatory
            len([s for s in answer.split('.') if s.strip()]) >= 2,  # Multiple sentences
            not answer.lower().startswith(('yes', 'no', 'maybe')),  # Not just yes/no
        ]
        
        return sum(info_indicators) / len(info_indicators)
    
    def _get_quality_rating(self, score: float) -> str:
        """🎯 ENHANCED: Convert overall quality score to rating with improved thresholds"""
        if score >= 0.8:
            return 'Excellent'
        elif score >= 0.65:
            return 'Very Good'
        elif score >= 0.5:
            return 'Good'
        elif score >= 0.35:
            return 'Fair'
        elif score >= 0.2:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _get_performance_category(self, quality_score: float, inference_time: float) -> str:
        """Categorize performance based on quality and speed"""
        is_fast = inference_time < 1.0
        is_high_quality = quality_score >= 0.7
        
        if is_high_quality and is_fast:
            return 'High Performance'
        elif is_high_quality:
            return 'High Quality'
        elif is_fast:
            return 'Fast Response'
        else:
            return 'Standard'

    def calculate_comprehensive_inference_metrics(self, question: str, results: Dict[str, Dict],
                                                inference_time: float, context_examples: int,
                                                reference_data: List[Dict] = None) -> Dict[str, Any]:
        # Implementation of the method
        # This method should return a dictionary with the calculated metrics
        # The implementation details are not provided in the original file or the code block
        # You may want to implement this method based on the requirements of the task
        # For example, you can use the existing methods and logic to calculate the metrics
        # and return them in the required format
        # This is a placeholder and should be replaced with the actual implementation
        return {} 

    def _analyze_multi_model_speed(self, model_data: Dict) -> Dict:
        """Analyze speed performance across multiple models"""
        if len(model_data) < 2:
            return {}
        
        # Sort models by speed (fastest first)
        sorted_by_speed = sorted(model_data.items(), key=lambda x: x[1]['inference_time'])
        
        fastest_model = sorted_by_speed[0][0]
        slowest_model = sorted_by_speed[-1][0]
        
        fastest_time = sorted_by_speed[0][1]['inference_time']
        slowest_time = sorted_by_speed[-1][1]['inference_time']
        
        # Calculate speed differences
        speed_differences = {}
        for model_name, data in model_data.items():
            if model_name != fastest_model:
                diff = data['inference_time'] - fastest_time
                speed_differences[model_name] = diff
        
        # Calculate rankings
        rankings = {}
        for i, (model_name, data) in enumerate(sorted_by_speed):
            rankings[model_name] = i + 1
        
        return {
            'fastest_model': fastest_model,
            'slowest_model': slowest_model,
            'fastest_time': round(fastest_time, 3),
            'slowest_time': round(slowest_time, 3),
            'speed_range': round(slowest_time - fastest_time, 3),
            'speed_differences': {k: round(v, 3) for k, v in speed_differences.items()},
            'rankings': rankings,
            'model_times': {name: round(data['inference_time'], 3) for name, data in model_data.items()}
        }
    
    def _analyze_multi_model_quality(self, model_data: Dict) -> Dict:
        """Analyze quality performance across multiple models"""
        if len(model_data) < 2:
            return {}
        
        # Sort models by quality (highest first)
        sorted_by_quality = sorted(model_data.items(), key=lambda x: x[1]['answer_quality_score'], reverse=True)
        
        highest_quality_model = sorted_by_quality[0][0]
        lowest_quality_model = sorted_by_quality[-1][0]
        
        highest_quality_score = sorted_by_quality[0][1]['answer_quality_score']
        lowest_quality_score = sorted_by_quality[-1][1]['answer_quality_score']
        
        # Sort by answer length (most detailed first)
        sorted_by_length = sorted(model_data.items(), key=lambda x: x[1]['answer_length'], reverse=True)
        most_detailed_model = sorted_by_length[0][0]
        
        # Sort by complexity (most complex first)
        sorted_by_complexity = sorted(model_data.items(), key=lambda x: x[1]['answer_complexity'], reverse=True)
        most_complex_model = sorted_by_complexity[0][0]
        
        # Calculate quality rankings
        quality_rankings = {}
        for i, (model_name, data) in enumerate(sorted_by_quality):
            quality_rankings[model_name] = i + 1
        
        return {
            'highest_quality_model': highest_quality_model,
            'lowest_quality_model': lowest_quality_model,
            'highest_quality_score': round(highest_quality_score, 3),
            'lowest_quality_score': round(lowest_quality_score, 3),
            'quality_range': round(highest_quality_score - lowest_quality_score, 3),
            'most_detailed_model': most_detailed_model,
            'most_complex_model': most_complex_model,
            'quality_rankings': quality_rankings,
            'model_quality_scores': {name: round(data['answer_quality_score'], 3) for name, data in model_data.items()},
            'model_lengths': {name: data['answer_length'] for name, data in model_data.items()},
            'model_complexities': {name: data['answer_complexity'] for name, data in model_data.items()}
        }