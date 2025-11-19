#!/usr/bin/env python3
"""
Ontology Manager - Manages multiple ontologies and their processing status

Features:
- Track multiple loaded ontologies
- Processing status (parsing, building, ready)
- Metadata management
- Query routing
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class ProcessingStatus(Enum):
    """Ontology processing status"""
    UPLOADING = "uploading"
    PARSING = "parsing"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"


class OntologyManager:
    """Manages multiple ontologies and their lifecycle"""
    
    def __init__(self, workspace_dir: str = "data/ontologies"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.workspace_dir / "ontologies_index.json"
        self.ontologies = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load ontologies index from disk"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save ontologies index to disk"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.ontologies, f, indent=2, ensure_ascii=False)
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def add_ontology(self, owl_file: str, name: Optional[str] = None) -> str:
        """
        Add new ontology to workspace
        
        Args:
            owl_file: Path to OWL file
            name: Optional custom name (auto-detected if not provided)
        
        Returns:
            ontology_id: Unique ID for this ontology
        """
        # Generate unique ID from file hash
        file_hash = self._compute_file_hash(owl_file)
        ontology_id = file_hash[:16]  # Use first 16 chars
        
        # Check if already exists
        if ontology_id in self.ontologies:
            return ontology_id
        
        # Create ontology directory
        onto_dir = self.workspace_dir / ontology_id
        onto_dir.mkdir(exist_ok=True)
        
        # Copy OWL file to workspace
        owl_dest = onto_dir / "ontology.owl"
        shutil.copy(owl_file, owl_dest)
        
        # Create initial metadata
        self.ontologies[ontology_id] = {
            'id': ontology_id,
            'name': name or Path(owl_file).stem,
            'original_filename': Path(owl_file).name,
            'file_hash': file_hash,
            'status': ProcessingStatus.UPLOADING.value,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'directory': str(onto_dir),
            'owl_file': str(owl_dest),
            'parsed_dir': None,
            'error_message': None,
            'metadata': None,
        }
        
        self._save_index()
        return ontology_id
    
    def update_status(self, ontology_id: str, status: ProcessingStatus, 
                     error_message: Optional[str] = None):
        """Update ontology processing status"""
        if ontology_id in self.ontologies:
            self.ontologies[ontology_id]['status'] = status.value
            self.ontologies[ontology_id]['updated_at'] = datetime.now().isoformat()
            if error_message:
                self.ontologies[ontology_id]['error_message'] = error_message
            self._save_index()
    
    def update_metadata(self, ontology_id: str, metadata: Dict):
        """Update ontology metadata after parsing/building"""
        if ontology_id in self.ontologies:
            self.ontologies[ontology_id]['metadata'] = metadata
            self.ontologies[ontology_id]['updated_at'] = datetime.now().isoformat()
            self._save_index()
    
    def get_ontology(self, ontology_id: str) -> Optional[Dict]:
        """Get ontology info by ID"""
        return self.ontologies.get(ontology_id)
    
    def list_ontologies(self, status: Optional[ProcessingStatus] = None) -> List[Dict]:
        """
        List all ontologies, optionally filtered by status
        
        Args:
            status: Filter by status (None = all)
        
        Returns:
            List of ontology metadata dicts
        """
        ontologies = list(self.ontologies.values())
        
        if status:
            ontologies = [o for o in ontologies if o['status'] == status.value]
        
        # Sort by creation time (newest first)
        ontologies.sort(key=lambda x: x['created_at'], reverse=True)
        
        return ontologies
    
    def get_ready_ontologies(self) -> List[Dict]:
        """Get list of ready-to-query ontologies"""
        return self.list_ontologies(status=ProcessingStatus.READY)
    
    def delete_ontology(self, ontology_id: str):
        """Delete ontology and all its files"""
        if ontology_id in self.ontologies:
            onto_dir = Path(self.ontologies[ontology_id]['directory'])
            if onto_dir.exists():
                shutil.rmtree(onto_dir)
            
            del self.ontologies[ontology_id]
            self._save_index()
    
    def get_ontology_dir(self, ontology_id: str) -> Optional[Path]:
        """Get directory path for ontology"""
        if ontology_id in self.ontologies:
            return Path(self.ontologies[ontology_id]['directory'])
        return None
    
    def get_parsed_dir(self, ontology_id: str) -> Optional[Path]:
        """Get parsed directory path for ontology"""
        onto_info = self.get_ontology(ontology_id)
        if onto_info and onto_info.get('parsed_dir'):
            return Path(onto_info['parsed_dir'])
        return None
    
    def set_parsed_dir(self, ontology_id: str, parsed_dir: str):
        """Set parsed directory after parsing complete"""
        if ontology_id in self.ontologies:
            self.ontologies[ontology_id]['parsed_dir'] = str(parsed_dir)
            self._save_index()


def main():
    """Test ontology manager"""
    manager = OntologyManager()
    
    print("Ontology Manager")
    print("=" * 80)
    print(f"Workspace: {manager.workspace_dir}")
    print(f"Loaded ontologies: {len(manager.ontologies)}")
    print()
    
    # List all ontologies
    if manager.ontologies:
        print("Registered ontologies:")
        for onto in manager.list_ontologies():
            print(f"  - {onto['name']} ({onto['id']}) - Status: {onto['status']}")
    else:
        print("No ontologies registered yet")


if __name__ == "__main__":
    main()
