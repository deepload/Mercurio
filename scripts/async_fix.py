#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patch script to fix async implementation in run_strategy_crypto_trader.py
"""

import sys
import os
import re
import asyncio

def patch_file(file_path):
    """Apply async fixes to the crypto trader script"""
    print(f"Patching file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add asyncio import if not present
    if 'import asyncio' not in content:
        content = content.replace(
            'import signal\nimport atexit',
            'import signal\nimport atexit\nimport asyncio'
        )
    
    # Make main function async
    content = content.replace(
        'def main():',
        'async def main():'
    )
    
    # Fix the trader.start() calls to use await
    content = re.sub(
        r'(\s+)if custom_duration:\s*\n\s+trader\.start\(custom_duration\)\s*\n\s+else:\s*\n\s+trader\.start\(\)',
        r'\1if custom_duration:\n\1    await trader.start(custom_duration)\n\1else:\n\1    await trader.start()',
        content
    )
    
    # Update run_crypto_trader to use asyncio.run
    content = re.sub(
        r'(\s+)try:\s*\n\s+success = main\(\)\s*\n\s+return success',
        r'\1try:\n\1    # Use asyncio.run to execute the async main function\n\1    success = asyncio.run(main())\n\1    return success',
        content
    )
    
    # Write the patched content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Successfully patched {file_path}")
    print("The script should now properly handle async functions")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default path
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'scripts', 'run_strategy_crypto_trader.py')
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    patch_file(file_path)
