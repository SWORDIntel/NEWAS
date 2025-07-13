#!/usr/bin/env python3
"""
NEMWAS Framework Structure Visualizer

Generates a visual tree structure of the framework with file statistics
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import json
import yaml

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

class FrameworkVisualizer:
    """Visualizes NEMWAS framework structure"""
    
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path.cwd()
        self.stats = {
            'directories': 0,
            'python_files': 0,
            'yaml_files': 0,
            'shell_scripts': 0,
            'markdown_files': 0,
            'total_files': 0,
            'total_lines': 0,
            'total_size': 0
        }
        
        # File type mappings for icons and colors
        self.file_icons = {
            '.py': ('🐍', Colors.GREEN),
            '.yaml': ('📄', Colors.YELLOW),
            '.yml': ('📄', Colors.YELLOW),
            '.sh': ('🔧', Colors.CYAN),
            '.md': ('📝', Colors.BLUE),
            '.txt': ('📃', Colors.DIM),
            '.json': ('📊', Colors.MAGENTA),
            'Dockerfile': ('🐳', Colors.BLUE),
            'Makefile': ('⚙️', Colors.RED),
        }
        
        # Directories to skip
        self.skip_dirs = {
            '__pycache__', '.git', '.pytest_cache', '.mypy_cache',
            'venv', 'env', '.venv', 'node_modules', '.idea', '.vscode',
            'htmlcov', 'build', 'dist', '*.egg-info'
        }
        
        # File patterns to skip
        self.skip_files = {
            '.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll',
            '.coverage', '.DS_Store', 'Thumbs.db'
        }
    
    def visualize(self, show_files: bool = True, max_depth: int = None):
        """Generate and print the framework structure"""
        print(f"{Colors.BOLD}{Colors.BLUE}NEMWAS Framework Structure{Colors.END}")
        print("=" * 60)
        print(f"Base: {self.base_path}")
        print("=" * 60)
        print()
        
        # Generate tree
        self._print_tree(self.base_path, "", True, show_files, max_depth, 0)
        
        # Print statistics
        self._print_statistics()
        
        # Print important files summary
        self._print_important_files()
    
    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped"""
        name = path.name
        
        # Skip hidden files/dirs (except .gitignore)
        if name.startswith('.') and name != '.gitignore':
            return True
        
        # Skip specified directories
        if path.is_dir() and name in self.skip_dirs:
            return True
        
        # Skip specified file types
        if path.is_file():
            for skip_ext in self.skip_files:
                if name.endswith(skip_ext):
                    return True
        
        return False
    
    def _get_file_info(self, path: Path) -> Tuple[str, str, int]:
        """Get file icon, color, and line count"""
        if not path.is_file():
            return '📁', Colors.BOLD + Colors.BLUE, 0
        
        # Check exact filename first
        if path.name in self.file_icons:
            icon, color = self.file_icons[path.name]
            lines = self._count_lines(path)
            return icon, color, lines
        
        # Then check extension
        ext = path.suffix
        if ext in self.file_icons:
            icon, color = self.file_icons[ext]
            lines = self._count_lines(path)
            return icon, color, lines
        
        # Default
        lines = self._count_lines(path)
        return '📄', Colors.DIM, lines
    
    def _count_lines(self, path: Path) -> int:
        """Count lines in a text file"""
        try:
            if path.suffix in ['.py', '.yaml', '.yml', '.sh', '.md', '.txt', '.json']:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return sum(1 for _ in f)
        except:
            pass
        return 0
    
    def _get_dir_summary(self, path: Path) -> Dict[str, int]:
        """Get summary of directory contents"""
        summary = {'py': 0, 'yaml': 0, 'total': 0}
        
        try:
            for item in path.iterdir():
                if self._should_skip(item):
                    continue
                    
                if item.is_file():
                    summary['total'] += 1
                    if item.suffix == '.py':
                        summary['py'] += 1
                    elif item.suffix in ['.yaml', '.yml']:
                        summary['yaml'] += 1
        except:
            pass
        
        return summary
    
    def _print_tree(self, path: Path, prefix: str, is_last: bool, 
                    show_files: bool, max_depth: int, current_depth: int):
        """Recursively print directory tree"""
        
        if self._should_skip(path):
            return
        
        # Check depth limit
        if max_depth is not None and current_depth > max_depth:
            return
        
        # Get file/dir info
        icon, color, lines = self._get_file_info(path)
        
        # Print item
        connector = "└── " if is_last else "├── "
        line_info = f" ({lines} lines)" if lines > 0 else ""
        
        if path.is_dir():
            # Get directory summary
            summary = self._get_dir_summary(path)
            if summary['total'] > 0:
                summary_str = f" [{summary['py']} py, {summary['total']} files]"
            else:
                summary_str = ""
            
            print(f"{prefix}{connector}{icon} {color}{path.name}/{Colors.END}{Colors.DIM}{summary_str}{Colors.END}")
            self.stats['directories'] += 1
        else:
            size_str = f" ({self._human_size(path.stat().st_size)})" if path.exists() else ""
            print(f"{prefix}{connector}{icon} {color}{path.name}{Colors.END}{Colors.DIM}{line_info}{size_str}{Colors.END}")
            self._update_stats(path, lines)
        
        # Process children
        if path.is_dir():
            try:
                # Get and sort children
                children = list(path.iterdir())
                children = [c for c in children if not self._should_skip(c)]
                
                # Sort: directories first, then alphabetically
                children.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
                
                # Skip files if not showing them
                if not show_files:
                    children = [c for c in children if c.is_dir()]
                
                # Process each child
                for i, child in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    extension = "    " if is_last else "│   "
                    self._print_tree(child, prefix + extension, is_last_child, 
                                   show_files, max_depth, current_depth + 1)
            except PermissionError:
                pass
    
    def _update_stats(self, path: Path, lines: int):
        """Update statistics"""
        self.stats['total_files'] += 1
        self.stats['total_lines'] += lines
        
        try:
            self.stats['total_size'] += path.stat().st_size
        except:
            pass
        
        ext = path.suffix
        if ext == '.py':
            self.stats['python_files'] += 1
        elif ext in ['.yaml', '.yml']:
            self.stats['yaml_files'] += 1
        elif ext == '.sh':
            self.stats['shell_scripts'] += 1
        elif ext == '.md':
            self.stats['markdown_files'] += 1
    
    def _human_size(self, size: int) -> str:
        """Convert size to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
    
    def _print_statistics(self):
        """Print framework statistics"""
        print()
        print("=" * 60)
        print(f"{Colors.BOLD}Framework Statistics{Colors.END}")
        print("=" * 60)
        
        stats_lines = [
            f"Directories:     {self.stats['directories']:,}",
            f"Total Files:     {self.stats['total_files']:,}",
            f"Python Files:    {self.stats['python_files']:,}",
            f"YAML Files:      {self.stats['yaml_files']:,}",
            f"Shell Scripts:   {self.stats['shell_scripts']:,}",
            f"Markdown Files:  {self.stats['markdown_files']:,}",
            f"Total Lines:     {self.stats['total_lines']:,}",
            f"Total Size:      {self._human_size(self.stats['total_size'])}"
        ]
        
        for line in stats_lines:
            print(f"  {line}")
        
        print()
    
    def _print_important_files(self):
        """Print summary of important files"""
        print(f"{Colors.BOLD}Key Files:{Colors.END}")
        
        important_files = [
            ("Entry Points", [
                "main.py",
                "quickstart.sh",
                "validate_framework.py"
            ]),
            ("Core Components", [
                "src/core/agent.py",
                "src/core/npu_manager.py",
                "src/core/react.py"
            ]),
            ("Configuration", [
                "config/default.yaml",
                "requirements.txt",
                "docker/docker-compose.yml"
            ]),
            ("Documentation", [
                "README.md",
                "examples/simple_example.py"
            ])
        ]
        
        for category, files in important_files:
            print(f"\n  {Colors.CYAN}{category}:{Colors.END}")
            for file in files:
                path = self.base_path / file
                if path.exists():
                    icon, color, lines = self._get_file_info(path)
                    status = f"{Colors.GREEN}✓{Colors.END}"
                    line_info = f" ({lines} lines)" if lines > 0 else ""
                else:
                    icon, color = '❌', Colors.RED
                    status = f"{Colors.RED}✗{Colors.END}"
                    line_info = " (missing)"
                
                print(f"    {status} {icon} {color}{file}{Colors.END}{Colors.DIM}{line_info}{Colors.END}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize NEMWAS framework structure")
    parser.add_argument("--no-files", action="store_true", help="Show only directories")
    parser.add_argument("--depth", type=int, help="Maximum depth to display")
    parser.add_argument("--path", type=str, help="Base path (default: current directory)")
    parser.add_argument("--export", type=str, help="Export structure to file (txt/json)")
    
    args = parser.parse_args()
    
    # Create visualizer
    base_path = Path(args.path) if args.path else Path.cwd()
    visualizer = FrameworkVisualizer(base_path)
    
    # Generate visualization
    visualizer.visualize(show_files=not args.no_files, max_depth=args.depth)
    
    # Export if requested
    if args.export:
        export_path = Path(args.export)
        
        if export_path.suffix == '.json':
            # Export as JSON
            data = {
                'base_path': str(visualizer.base_path),
                'statistics': visualizer.stats,
                'generated': str(Path.cwd())
            }
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            # Export as text (redirect stdout)
            import contextlib
            with open(export_path, 'w') as f:
                with contextlib.redirect_stdout(f):
                    visualizer.visualize(show_files=not args.no_files, max_depth=args.depth)
        
        print(f"\nStructure exported to: {export_path}")

if __name__ == "__main__":
    main()
