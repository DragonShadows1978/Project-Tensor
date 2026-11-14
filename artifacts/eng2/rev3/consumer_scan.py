"""Independent, conservative static consumer inventory. No consumer code runs.

Prior art: Python project's ast syntax traversal; data-flow/constant propagation
in the tradition of Kildall (1973), unverified -- lead to check 'Kildall 1973
data flow analysis'. Taken: syntax inspection and monotone alias propagation.
Review-specific: seven-entry dtype boundary, complete tracked-Python operand
listing, role census, conservative unresolved records and snapshot receipts.
This is a static inventory, not proof over arbitrary dynamic Python behavior.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path.cwd()
OUT = ROOT / 'artifacts/eng2/rev3'
SCRATCH = Path(os.environ['TMPDIR']) / 'consumer_sources'
SUPPORTED = {'float32','float16','bfloat16','int64','bool','uint8'}
ENTRIES = {'tensor','zeros','ones','randn','rand','parameter','_causal_mask'}
COMMANDS = []


def git(repo, *args):
    argv = ['git','-C',str(repo),*args]
    r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    COMMANDS.append(dict(argv=argv, exit=r.returncode, stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),
                         stderr=r.stderr.decode(errors='replace')))
    if r.returncode:
        return None
    return r.stdout


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return base + '.' + node.attr if base else ''
    return ''


def role(path):
    parts = Path(path).parts
    if 'artifacts' in parts:
        return 'receipt'
    if any(p in {'test','tests','p1t_tests'} for p in parts) or Path(path).name.startswith('test_'):
        return 'test'
    if any(p in {'script','scripts','tools'} for p in parts):
        return 'script'
    return 'product'


def resolve(name, aliases):
    if not name:
        return ''
    parts = name.split('.')
    for n in range(len(parts),0,-1):
        prefix = '.'.join(parts[:n])
        if prefix in aliases:
            return aliases[prefix] + ('.' + '.'.join(parts[n:]) if n < len(parts) else '')
    return name


def imports_and_aliases(tree, module):
    aliases = {}
    links = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                aliases[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            if n.level:
                base = module.split('.')[:-n.level]
                mod = '.'.join(base + ([mod] if mod else []))
            for a in n.names:
                if a.name == '*':
                    links.append(('*',mod))
                else:
                    aliases[a.asname or a.name] = mod + '.' + a.name
        elif isinstance(n, (ast.Assign,ast.AnnAssign)) and n.value is not None:
            targets = n.targets if isinstance(n,ast.Assign) else [n.target]
            value = dotted(n.value)
            if isinstance(n.value,ast.Call) and dotted(n.value.func) in {'importlib.import_module','__import__','import_module'}:
                if n.value.args and isinstance(n.value.args[0],ast.Constant) and isinstance(n.value.args[0].value,str):
                    value = n.value.args[0].value
            if value:
                for t in targets:
                    if dotted(t):
                        links.append((dotted(t), value))
    return aliases, links


def gather_context(tree):
    """Retain lexical scope, parameters/defaults, assignments, and in-file calls."""
    assignments = defaultdict(list)
    parameters = {}
    calls_to = defaultdict(list)
    node_scope = {}
    scope_parent = {'<module>':None}
    function_names = {}

    def walk(node, scope):
        node_scope[id(node)] = scope
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda)):
            name = getattr(node,'name','<lambda>')
            new = scope + '/' + name + '@' + str(node.lineno)
            scope_parent[new] = scope
            function_names[(scope,name)] = new
            aa = node.args
            pos = aa.posonlyargs + aa.args
            defaults = [None]*(len(pos)-len(aa.defaults)) + list(aa.defaults)
            params = {a.arg: (i, d) for i,(a,d) in enumerate(zip(pos,defaults))}
            params.update({a.arg:(None,d) for a,d in zip(aa.kwonlyargs,aa.kw_defaults)})
            parameters[new] = params
            for child in ast.iter_child_nodes(node):
                walk(child, new)
            return
        if isinstance(node,(ast.Assign,ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target,ast.Name):
                    assignments[(scope,target.id)].append(node.value)
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name):
            if isinstance(node.iter,(ast.Tuple,ast.List,ast.Set)):
                assignments[(scope,node.target.id)].extend(node.iter.elts)
        if isinstance(node,ast.Call):
            calls_to[dotted(node.func)].append(node)
        for child in ast.iter_child_nodes(node):
            walk(child,scope)
    walk(tree,'<module>')
    return assignments, parameters, calls_to, node_scope, scope_parent, function_names


def classify(expr, scope, context, aliases, seen=()):
    assigns, params, calls, node_scopes, parents, functions = context
    if expr is None:
        return {'supported:<default float32>'}
    if isinstance(expr,ast.Constant):
        if isinstance(expr.value,str):
            return {('supported:' if expr.value in SUPPORTED else 'unsupported:') + repr(expr.value)}
        return {'non-string:' + repr(expr.value)}
    if isinstance(expr,(ast.List,ast.Tuple,ast.Dict,ast.Set,ast.Lambda)):
        return {'non-string:' + ast.unparse(expr)}
    if isinstance(expr,ast.IfExp):
        return classify(expr.body,scope,context,aliases,seen) | classify(expr.orelse,scope,context,aliases,seen)
    if isinstance(expr,ast.BoolOp) and isinstance(expr.op,ast.Or):
        values = set()
        for i,e in enumerate(expr.values):
            v = classify(e,scope,context,aliases,seen)
            # A false None does not reach the dtype when followed by a fallback.
            values |= (v-{'non-string:None','non-string:False',"unsupported:''"}) if i<len(expr.values)-1 else v
        return values or {'unresolved:or-expression'}
    if isinstance(expr,ast.Name):
        name = resolve(expr.id,aliases)
        if name in {'float','int','str','bool','object'}:
            return {'non-string:type ' + name}
        key = (scope,expr.id)
        if key in seen:
            return {'unresolved:recursive ' + expr.id}
        s = scope
        while s is not None:
            if (s,expr.id) in assigns:
                return set().union(*(classify(v,s,context,aliases,seen+(key,)) for v in assigns[(s,expr.id)]))
            if expr.id in params.get(s,{}):
                index, default = params[s][expr.id]
                function = s.rsplit('/',1)[-1].split('@')[0]
                supplied = []
                for call in calls.get(function,[]):
                    value = next((k.value for k in call.keywords if k.arg==expr.id),None)
                    if value is None and index is not None and index<len(call.args):
                        value = call.args[index]
                    if value is not None:
                        supplied.append(classify(value,node_scopes[id(call)],context,aliases,seen+(key,)))
                    elif default is not None:
                        supplied.append(classify(default,s,context,aliases,seen+(key,)))
                # A default alone says nothing about callers outside this file.
                if supplied:
                    return set().union(*supplied)
                return {'unresolved:parameter ' + expr.id}
            s = parents[s]
        if name.startswith('numpy.') and name.rsplit('.',1)[-1] in {'float16','float32','float64','int32','int64','bool_','uint8'}:
            return {'non-string:' + name}
        return {'unresolved:name ' + name}
    if isinstance(expr,ast.Attribute):
        name = resolve(dotted(expr),aliases)
        if name.startswith('numpy.') and name.rsplit('.',1)[-1] in {'float16','float32','float64','int32','int64','bool_','uint8'}:
            return {'non-string:' + name}
        return {'unresolved:attribute ' + ast.unparse(expr)}
    if isinstance(expr,ast.Call):
        name = resolve(dotted(expr.func),aliases)
        if name in {'numpy.dtype','type','numpy.float16','numpy.float32','numpy.float64'}:
            return {'non-string:' + ast.unparse(expr)}
        return {'unresolved:call ' + ast.unparse(expr)}
    return {'unresolved:expression ' + ast.unparse(expr)}


def main():
    result = dict(method='all tracked .py at pinned HEAD, git-show source, independent AST aliases and scoped value tracing',
                  supported=sorted(SUPPORTED), repos=[], parse_failures=[], sites=[], candidate_calls_unbound=[])
    for reponame in ['Project-Voice','GRAPA-Native-LLM','GraftRepository']:
        repo = Path('/mnt/ForgeRealm') / reponame
        head = git(repo,'show','-s','--format=%H','HEAD').decode().strip()
        paths = git(repo,'ls-files','-z','--','*.py').decode().split('\0')
        paths = [p for p in paths if p and 'WorkTree' not in Path(p).parts]
        modules = {}
        listing = []
        for path in paths:
            raw = git(repo,'show',head + ':' + path)
            if raw is None:
                result['parse_failures'].append(dict(repo=reponame,file=path,reason='not at pinned HEAD'))
                continue
            target = SCRATCH / reponame / path
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
            try:
                tree = ast.parse(raw,filename=path)
            except (SyntaxError,ValueError) as exc:
                result['parse_failures'].append(dict(repo=reponame,file=path,reason=str(exc)))
                continue
            module = path[:-3].replace('/','.')
            if module.endswith('.__init__'):
                module = module[:-9]
            aliases,links = imports_and_aliases(tree,module)
            modules[module] = dict(path=path,aliases=aliases,links=links)
            listing.append(dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),role=role(path)))
        # Export lookup follows ordinary imports/re-exports, and assignment aliases.
        exports = {}
        for iteration in range(20):
            changes = 0
            for module,data in modules.items():
                aliases = data['aliases']
                for name,value in list(aliases.items()):
                    resolved = exports.get(value,value)
                    if resolved != value:
                        aliases[name]=resolved
                        changes+=1
                for name,value in data['links']:
                    if name=='*':
                        for key,val in list(exports.items()):
                            if key.startswith(value+'.') and '.' not in key[len(value)+1:]:
                                local=key[len(value)+1:]
                                if val.startswith('tensor_cuda') and aliases.get(local)!=val:
                                    aliases[local]=val; changes+=1
                        continue
                    resolved=resolve(value,aliases)
                    if resolved.startswith('tensor_cuda') and aliases.get(name)!=resolved:
                        aliases[name]=resolved; changes+=1
                for name,value in aliases.items():
                    key=module+'.'+name
                    if value.startswith('tensor_cuda') and exports.get(key)!=value:
                        exports[key]=value; changes+=1
            if not changes:
                break
        local_rows=[]
        for module,data in modules.items():
            path=data['path']
            raw=(SCRATCH/reponame/path).read_bytes()
            tree=ast.parse(raw,filename=path)
            aliases=data['aliases']
            ctx=gather_context(tree)
            for node in ast.walk(tree):
                if not isinstance(node,ast.Call):
                    continue
                func=dotted(node.func)
                resolved=resolve(func,aliases)
                last=resolved.rsplit('.',1)[-1]
                if last not in ENTRIES:
                    continue
                dtype=next((k.value for k in node.keywords if k.arg=='dtype'),None)
                kwargs=any(k.arg is None for k in node.keywords)
                if last=='_causal_mask' and dtype is None and len(node.args)>=4:
                    dtype=node.args[3]
                row=dict(repo=reponame,head=head,file=path,line=node.lineno,col=node.col_offset,
                         role=role(path),function=func,resolved=resolved,
                         entry=last,argument=ast.unparse(dtype) if dtype is not None else ('**kwargs' if kwargs else '<default float32>'),
                         source=ast.get_source_segment(raw.decode(errors='replace'),node))
                if not resolved.startswith('tensor_cuda.'):
                    # Every potentially relevant unbound receiver is retained for review.
                    if dtype is not None or kwargs:
                        result['candidate_calls_unbound'].append(row)
                    continue
                vals=classify(dtype,ctx[3][id(node)],ctx,aliases) if dtype is not None else ({'unresolved:**kwargs'} if kwargs else {'supported:<default float32>'})
                row['values']=sorted(vals)
                if any(x.startswith(('unsupported:','non-string:')) for x in vals):
                    row['classification']='potentially-newly-refused'
                elif any(x.startswith('unresolved:') for x in vals):
                    row['classification']='unresolved'
                else:
                    row['classification']='supported'
                local_rows.append(row)
        result['sites'].extend(local_rows)
        result['repos'].append(dict(repo=reponame,head=head,tracked_python=len(paths),parsed=len(listing),
                                    source_files=listing,alias_iterations=iteration+1,
                                    counts=dict(Counter(r['classification'] for r in local_rows)),
                                    roles=dict(Counter(r['role'] for r in local_rows)),
                                    entries=dict(Counter(r['entry'] for r in local_rows))))
        print(reponame,head,len(paths),'tracked Python;',len(local_rows),'changed-entry sites;',Counter(r['classification'] for r in local_rows),flush=True)
    result['newly_refused']=[r for r in result['sites'] if r['classification']=='potentially-newly-refused']
    result['unresolved']=[r for r in result['sites'] if r['classification']=='unresolved']
    (OUT/'Q3_consumer_scan.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'Q3_git_commands.json').write_text(json.dumps(COMMANDS,indent=2)+'\n')
    print('potentially refused',len(result['newly_refused']),'unresolved',len(result['unresolved']),'parse failures',len(result['parse_failures']),flush=True)


if __name__=='__main__':
    main()
