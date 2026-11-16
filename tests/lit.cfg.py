# -*- Python -*-

import os
import lit.formats

# name: The name of this test suite.
config.name = 'jlm'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(execute_external=True)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# project_root: The root of the repository.
project_root = os.path.abspath(os.path.join(config.test_source_root, '..'))

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(project_root, 'build', 'tests')

# Helper to parse Makefile.config if present
makefile_config_path = os.path.join(project_root, 'Makefile.config')
makefile_vars = {}
if os.path.isfile(makefile_config_path):
    with open(makefile_config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('export '):
                line = line[len('export '):].strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                makefile_vars[k.strip()] = v.strip()

# Pass through essential environment variables
for env_var in ['PATH', 'LD_LIBRARY_PATH', 'TMP', 'TEMP', 'TMPDIR', 'HOME', 'USER']:
    if env_var in os.environ:
        config.environment[env_var] = os.environ[env_var]

# If LD_LIBRARY_PATH was configured in Makefile.config, prepend/append it
if 'LD_LIBRARY_PATH' in makefile_vars and makefile_vars['LD_LIBRARY_PATH']:
    extra_ld = makefile_vars['LD_LIBRARY_PATH']
    current_ld = config.environment.get('LD_LIBRARY_PATH', '')
    if current_ld:
        config.environment['LD_LIBRARY_PATH'] = f"{extra_ld}:{current_ld}"
    else:
        config.environment['LD_LIBRARY_PATH'] = extra_ld

# Locate binaries for substitutions
jlc_path = os.path.join(project_root, 'build', 'jlc')
jlm_opt_path = os.path.join(project_root, 'build', 'jlm-opt')

config.substitutions.append(('%jlc', jlc_path))
config.substitutions.append(('%jlm-opt', jlm_opt_path))
