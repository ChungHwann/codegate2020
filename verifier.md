# verifier

## code
문제 코드는 실행되는 prob.py와 모듈화된 파이썬 코드 4개로 이루어져있다. 결론적으로 prob.py는 사용자 입력을 받아 명령을 수행하는 인터프리터와 같은 동작을 한다. 이는 ply라는 파이썬 모듈을 통해 구현된다.
1. prob.py
    + 입력(명령)을 받은 후 syntax 검사 -> **a_interp -> interp**
    + a_interp는 interp 전에 수행되는 준비과정같은 느낌이었음. 예를 들어 어떤 변수가 가질 수 있는 값의 범위를 a_interp에서 정하고, 그 변수의 실제 값은 interp에서 정해지는 식. 
```python
if __name__ == '__main__':
    code = input('> ')
    try:
        ast = yacc.parse(code)
    except SyntaxError as syntaxE:
        print('SyntaxError: {}'.format(syntaxE))
        exit()

    try:
        ast.a_interp({})
    except Exception as e:
        print('Error: {}'.format(e))
        exit()

    try:
        ast.interp({})
    except Exception as e:
        print('Error: {}'.format(e))
        exit()
```
2. parser.py
    + 사용자 입력(명령)을 분석, 파싱
    + 명령어 형식을 확인할 수 
```python
...
def p_goal(p):
    """goal : comm"""
    p[0] = p[1]

def p_comm_1(p):
    """comm : COMMA"""
    p[0] = ast.Skip()

def p_comm_2(p):
    """comm : VAR ASSIGN expr"""
    p[0] = ast.Assign(ast.Var(p[1]), p[3])
...

comparison_dict = {
        '<': ast.Lt,
        '<=': ast.Le,
        '==': ast.Eq,
        '!=': ast.Ne,
        '>': ast.Gt,
        '>=': ast.Ge
}

def p_cond(p):
    """cond : VAR LT NUM
            | VAR LE NUM
            | VAR EQ NUM
            | VAR NE NUM
            | VAR GT NUM
            | VAR GE NUM"""
    p[0] = comparison_dict[p[2]](ast.Var(p[1]), ast.Num(p[3]))

def p_expr_1(p):
    """expr : LPAREN expr RPAREN"""
    p[0] = p[2]
...

binaryop_dict = {
        '+': ast.Add,
        '-': ast.Sub,
        '*': ast.Mul,
}

def p_expr_5(p):
    """expr : expr PLUS expr
            | expr MINUS expr
            | expr TIMES expr"""
    p[0] = binaryop_dict[p[2]](p[1], p[3])

def p_error(p):
    if p is None:
        raise SyntaxError("invalid syntax")
    raise SyntaxError("invalid syntax at '{}'".format(p.value))
...
```
3. ast.py
    + 명령어 수행에 대한 함수, 객체 모듈
    + 각 명령어가 수행될 때 (a_interp, interp) 어떻게 동작하는지 확인
```python
...
class Assign(Comm):
    def __init__(self, var, expr):
        self.var = var
        self.expr = expr

    def a_interp(self, env):
        env[self.var.name] = self.expr.a_interp(env)
        return env

    def interp(self, env):
        env[self.var.name] = self.expr.interp(env)
        return env

class Seq(Comm):
...

class IfElse(Comm):
...

class While(Comm):
...

class Print(Comm):
...
```
4. lexer.py
    + 토큰 정의
```python
tokens = [
    'VAR',
    'NUM',

    'PLUS',
    'MINUS',
    'TIMES',
...
    'RBRACK',
    'LBRACE',
    'RBRACE',

    'PRINT',
    'RANDOM',
]

t_ignore = ' \t\v\n\f'

t_VAR = r'[a-zA-Z_][a-zA-Z_0-9]*'

def t_NUM(t):
    r'[-+]?\d+'
    t.value = int(t.value)
    return t

t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
...
t_RBRACK = r'\]'
t_LBRACE = r'\{'
t_RBRACE = r'\}'

t_PRINT = r'!'
t_RANDOM = r'~'

def t_error(t):
    print("Illegal character '{}'".format(t.value[0]))
    t.lexer.skip(1)

lexer = lex.lex()
```
5. domain.py
    + 어떤 연산 이후 그 값의 최솟값, 최댓값 수정
```python
class Interval(object):
    def __init__(self, infimum, supremum):
        assert infimum <= supremum
        self.infimum = infimum
        self.supremum = supremum

    def __add__(self, other):
        infimum = self.infimum + other.infimum
        supremum = self.supremum + other.supremum
        return Interval(infimum, supremum)
...
    def __repr__(self):
        return '[{}, {}]'.format(self.infimum, self.supremum)
```
-----------------------
## Analysis
### Objective
+ ast.py, lexer.py, parser.py 의 존재 이유는 명확해 보이는데, domain.py 는 어떤 값의 최솟값과 최댓값을 수정해주는 역할로 굳이 왜 있어야 하나 하는 생각이 먼저 들었다.
+ ast.py 에서, Print 객체 내부에 이런 코드가 있다.
```python
class Print(Comm):
    def __init__(self, expr):
        self.expr = expr

    def a_interp(self, env):
        a_val = self.expr.a_interp(env)
        if a_val.infimum < 0:
            raise ValueError("print domain error")
        return env

    def interp(self, env):
        value = self.expr.interp(env)

        if value < 0:
            with open('./flag') as f:
                print(f.read())
        print(value)
        return env
```
+ a_interp 과정에서 self.expr.a_interp(env)의 최솟값(infimum)이 0보다 커야 Error을 피할 수 있다. 
+ flag를 출력시키려면 interp 과정에서 value가 0보다 작아야하는데, 문제는 value의 최솟값이 0보다 작으면 a_interp 과정에서 Error이 난다는 것이다. 즉 최솟값은 0보다 큰데, 실제 그 값은 0보다 작은 값을 print 해야한다.

>*따라서 우리의 목표는 print명령 실행과정에서, a_interp 시 value의 infimum은 0보다 크게, interp 시 value 값은 0보다 작게끔 하는 것이다.*

+ 이를 위해 주목해야 할 것은 두 가지다.
>   1. **infimum이 정해지는 원리**
>   2. **a_interp, interp 차이**
+ 추가적으로 **모든 명령의 동작 과정을 이해**해야 원하는 명령을 줌으로써 flag를 얻을 수 있을 것이다. 이에 상당히 많은 시간이 필요했다.
### infimum, supremum
infimum과 supremum이 정해지는 과정은 domain.py 를 통해 알 수 있다.
```python
class Interval(object):
    def __init__(self, infimum, supremum):
        assert infimum <= supremum
        self.infimum = infimum
        self.supremum = supremum

    def __add__(self, other):
        infimum = self.infimum + other.infimum
        supremum = self.supremum + other.supremum
        return Interval(infimum, supremum)

    def __sub__(self, other):
        infimum = self.infimum - other.supremum
        supremum = self.supremum - other.infimum
        return Interval(infimum, supremum)

    def __mul__(self, other):
        candidate = [self.infimum * other.infimum, self.infimum * other.supremum, self.supremum * other.infimum, self.supremum * other.supremum]
        return Interval(min(candidate), max(candidate))

    def __or__(self, other):
        return Interval(min(self.infimum, other.infimum), max(self.supremum, other.supremum))
```
+ 생성시 infimum <= supremum 인지 검사
+ +,-,*,or 연산 시 infimum, supremum을 정하는 원리는 수학적으로 검증된 원리
```python
    def __repr__(self):
        return '[{}, {}]'.format(self.infimum, self.supremum)

    def widen(self, other):
        w_infimum = self.infimum if self.infimum <= other.infimum else -inf
        w_supremum = self.supremum if other.supremum <= self.supremum else inf
        return Interval(w_infimum, w_supremum)
```
+ 따라서 우리가 주목할 부분은 이곳. self.infimum이 other.infimum보다 작거나 같으면 self.infimum을 -inf로 확장함. supremum도 마찬가지. 이 widen 함수가 어떻게 쓰이는 지 확인해볼 필요있어보인다.
### a_interp, interp
+ 굳이 a_interp, interp 두 번에 걸쳐서 명령어를 분석, 수행하는 이유가 의문이었다. 
+ 앞서 언급했던 대로 a_interp는 interp를 위한 준비과정 같은 느낌이었다. 
+ ast.py 에서 정의된 명령들 마다 a_interp, interp 과정에서 동작이 다른 명령들이 존재한다.
+ 이를 면밀히 분석하고자 명령마다 a_interp, interp 에서 수행되는 과정을 출력하도록 ast.py 를 수정 후 실행해보았다.
```python
class Assign(Comm):
    def __init__(self, var, expr):
        self.var = var
        self.expr = expr

    def a_interp(self, env):
        env[self.var.name] = self.expr.a_interp(env)
        print('a) Assign) env: '+str(env))
        return env

    def interp(self, env):
        env[self.var.name] = self.expr.interp(env)
        print('Assign) env: '+str(env))
        return env
...
class IfElse(Comm):
    def __init__(self, cond, comm1, comm2):
        self.cond = cond
        self.comm1 = comm1
        self.comm2 = comm2

    def a_interp(self, env):
        tenv, fenv = self.cond.a_interp(env)
        print('a) if) tenv, fenv: '+str(tenv)+', '+str(fenv))
        if tenv is not None:
            tenv = self.comm1.a_interp(tenv)
        if fenv is not None:
            fenv = self.comm2.a_interp(fenv)

        return env_join(tenv, fenv)

    def interp(self, env):
        cond = self.cond.interp(env)
        print('if) cond: '+str(cond))
        if cond:
            env = self.comm1.interp(env)
        else:
            env = self.comm2.interp(env)

        return env
...
```
*예시*
```
hanch@ubuntu:~/Documents/cykor/codegate2020/verifier$ python3 prob.py
> a=1~10;a>5?{!a}:{!1} 
a) Assign) env: {'a': [1, 10]}
a) if) tenv, fenv: {'a': [6, 10]}, {'a': [1, 5]}
a) print) env: {'a': [6, 10]}
a) print) a_val: [6, 10]
a) print) a_val.infimum: 6
a) print) env: {'a': [1, 5]}
a) print) a_val: [1, 1]
a) print) a_val.infimum: 1
a) Seq) env: {'a': [1, 10]}
Assign) env: {'a': 5}
if) cond: False
value: 1
1
Seq) env: {'a': 5}
```
1. *Assign의 경우 a_interp에서 변수 a가 가질 수 있는 값의 범위가 정해진다.*
2. *If의 경우 a_interp에서 각각 true, false이기 위한 범위가 정해진다.*
3. *Print의 경우 a_interp에서 출력되는 변수의 범위를 검사한다*
4. *interp에서 (Assign)a는 5로 정해지고 (If)If문의 결과는 False로 정해지며 명령에 따라 (Print)1이 출력된다.*
+ 여러 명령을 시도해보았고, 그 결과와 ast.py 의 코드를 분석하며 a_interp, interp 과정에서 차이가 있는 객체들을 분석했다.

-------------------
## Vulnerability
### While
ast.py 의 객체들 중 a_interp, interp 과정에서 차이가 있는 객체를 살펴보다가 While 객체 내에서 취약점이 발견되었다.
```python
class While(Comm):
    def __init__(self, cond, comm):
        self.cond = cond
        self.comm = comm

    def a_interp(self, env):
        init_env = deepcopy(env)

        for i in range(3): # check only 3 times !
            tenv, _ = self.cond.a_interp(env)
            if tenv is not None:
                tenv = self.comm.a_interp(tenv)
            env = env_join(env, tenv)

        tenv, _ = self.cond.a_interp(env)
        if tenv is not None:
            tenv = self.comm.a_interp(tenv)
        env = env_widen(env, tenv) # then widen !

        tenv, _ = self.cond.a_interp(env)
        if tenv is not None:
            tenv = self.comm.a_interp(tenv)
        env = env_join(init_env, tenv)
        _, fenv = self.cond.a_interp(env)

        if fenv is None:
            raise RuntimeError("loop analysis error")
    
        return fenv

    def interp(self, env):
        global loop_count
        cond = self.cond.interp(env)

        while cond: # do real loop
            env = self.comm.interp(env)
            cond = self.cond.interp(env)
            loop_count += 1
            if loop_count > 10000:
                raise RuntimeError("infinite loop error")
        loop_count = 0

        return env
```
+ While의 경우 a_interp에서 실제 반복문을 전부 수행하지 않고, 3번만 수행한 뒤 env_widen을 호출한다. 
+ 그러나 interp에서는 실제 반복문이 전부 수행된다.

*이는 무한루프 시 변수의 infimum, supremum을 특정할 수 없는 문제를 해결하고, 인터프리터의 속도를 향상시키기 위한 조치임으로 보여진다.*

+ a_interp에서 조건문이 각각 true, false가 되는 조건(변수의 범위)를 tenv, fenv로 저장한다. 
+ While은 조건문이 false일 때 끝나므로 While이후 변수의 범위를 fenv로 설정한다(return fenv). 

```
hanch@ubuntu:~/Documents/cykor/codegate2020/verifier$ python3 prob.py
> a=1;one=1;[a<10{a=a+one}]
a) Assign) env: {'a': [1, 1]}
a) Assign) env: {'a': [1, 1], 'one': [1, 1]}
a) Assign) env: {'a': [2, 2], 'one': [1, 1]}
a) While) i, env: 0, {'one': [1, 1], 'a': [1, 2]}
a) Assign) env: {'one': [1, 1], 'a': [2, 3]}
a) While) i, env: 1, {'one': [1, 1], 'a': [1, 3]}
a) Assign) env: {'one': [1, 1], 'a': [2, 4]}
a) While) i, env: 2, {'one': [1, 1], 'a': [1, 4]}
a) While) after) tenv: {'one': [1, 1], 'a': [1, 4]}
a) Assign) env: {'one': [1, 1], 'a': [2, 5]}
a) While) after) tenv: {'one': [1, 1], 'a': [2, 5]}
a) While) after) tenv: {'one': [1, 1], 'a': [1, 9]}
a) Assign) env: {'one': [1, 1], 'a': [2, 10]}
a) While) after) tenv: {'one': [1, 1], 'a': [2, 10]}
a) While) after) fenv: {'one': [1, 1], 'a': [10, 10]}
a) Seq) env: {'one': [1, 1], 'a': [10, 10]}
```
+ 이때 fenv is none일 경우 무한루프로 판정한다.  
```
hanch@ubuntu:~/Documents/cykor/codegate2020/verifier$ python3 prob.py
> a=1;one=1;[a>0{a<10?{a=a+one}:{.}}]
a) Assign) env: {'a': [1, 1]}
a) Assign) env: {'a': [1, 1], 'one': [1, 1]}
a) if) tenv, fenv: {'a': [1, 1], 'one': [1, 1]}, None
a) Assign) env: {'a': [2, 2], 'one': [1, 1]}
a) While) i, env: 0, {'a': [1, 2], 'one': [1, 1]}
a) if) tenv, fenv: {'a': [1, 2], 'one': [1, 1]}, None
a) Assign) env: {'a': [2, 3], 'one': [1, 1]}
a) While) i, env: 1, {'a': [1, 3], 'one': [1, 1]}
a) if) tenv, fenv: {'a': [1, 3], 'one': [1, 1]}, None
a) Assign) env: {'a': [2, 4], 'one': [1, 1]}
a) While) i, env: 2, {'a': [1, 4], 'one': [1, 1]}
a) While) after) tenv: {'a': [1, 4], 'one': [1, 1]}
a) if) tenv, fenv: {'a': [1, 4], 'one': [1, 1]}, None
a) Assign) env: {'a': [2, 5], 'one': [1, 1]}
a) While) after) tenv: {'a': [2, 5], 'one': [1, 1]}
a) While) after) tenv: {'a': [1, inf], 'one': [1, 1]}
a) if) tenv, fenv: {'a': [1, 9], 'one': [1, 1]}, {'a': [10, inf], 'one': [1, 1]}
a) Assign) env: {'a': [2, 10], 'one': [1, 1]}
a) While) after) tenv: {'a': [2, inf], 'one': [1, 1]}
a) While) after) fenv: None
Error: loop analysis error
```
>*env_widen 때문에, 실제 a의 범위는 1 ~ 10이지만, a_interp의 결과 a의 범위는 1 ~ inf이다.*

### widen
>다음과 같은 코드를 prob.py에 명령으로 넣어준 경우를 생각해보자. 
```python
a = 1
while a<10:
    a += 1
```
While에서 a_interp시 3번의 반복 후 env_widen을 하면서 이때 a의 범위는 [1,inf]가 된다. 그런데 return fenv때문에 While이 종료되면 a의 범위는 [10,10]이 된다.
>이때 while안에 a가 아닌 다른 변수가 있는 경우를 생각해보자.
```python
a = 1
b = 1
while a<10:
    a += 1
    b += 1
```
조건문에 b의 범위에 대한 언급은 없다. 따라서 return fenv가 b의 범위에 영향을 주지않으므로, b는 while 종료 후에도 env_widen 이후의 범위인 [1,inf]가 유지된다.
```
hanch@ubuntu:~/Documents/cykor/codegate2020/verifier$ python3 prob.py 
> a=1;b=1;one=1;[a<10{a=a+one;b=b+one}]
a) Assign) env: {'a': [1, 1]}
a) Assign) env: {'a': [1, 1], 'b': [1, 1]}
a) Assign) env: {'a': [1, 1], 'b': [1, 1], 'one': [1, 1]}
a) Assign) env: {'a': [2, 2], 'b': [1, 1], 'one': [1, 1]}
a) Assign) env: {'a': [2, 2], 'b': [2, 2], 'one': [1, 1]}
a) Seq) env: {'a': [2, 2], 'b': [2, 2], 'one': [1, 1]}
a) While) i, env: 0, {'one': [1, 1], 'b': [1, 2], 'a': [1, 2]}
a) Assign) env: {'one': [1, 1], 'b': [1, 2], 'a': [2, 3]}
a) Assign) env: {'one': [1, 1], 'b': [2, 3], 'a': [2, 3]}
a) Seq) env: {'one': [1, 1], 'b': [2, 3], 'a': [2, 3]}
a) While) i, env: 1, {'one': [1, 1], 'a': [1, 3], 'b': [1, 3]}
a) Assign) env: {'one': [1, 1], 'a': [2, 4], 'b': [1, 3]}
a) Assign) env: {'one': [1, 1], 'a': [2, 4], 'b': [2, 4]}
a) Seq) env: {'one': [1, 1], 'a': [2, 4], 'b': [2, 4]}
a) While) i, env: 2, {'one': [1, 1], 'a': [1, 4], 'b': [1, 4]}
a) While) after) tenv: {'one': [1, 1], 'a': [1, 4], 'b': [1, 4]}
a) Assign) env: {'one': [1, 1], 'a': [2, 5], 'b': [1, 4]}
a) Assign) env: {'one': [1, 1], 'a': [2, 5], 'b': [2, 5]}
a) Seq) env: {'one': [1, 1], 'a': [2, 5], 'b': [2, 5]}
a) While) after) tenv: {'one': [1, 1], 'a': [2, 5], 'b': [2, 5]}

after widen: {'one': [1, 1], 'a': [1, inf], 'b': [1, inf]}

a) While) after) tenv: {'one': [1, 1], 'a': [1, 9], 'b': [1, inf]}
a) Assign) env: {'one': [1, 1], 'a': [2, 10], 'b': [1, inf]}
a) Assign) env: {'one': [1, 1], 'a': [2, 10], 'b': [2, inf]}
a) Seq) env: {'one': [1, 1], 'a': [2, 10], 'b': [2, inf]}
a) While) after) tenv: {'one': [1, 1], 'a': [2, 10], 'b': [2, inf]}
a) While) after) fenv: {'one': [1, 1], 'b': [1, inf], 'a': [10, 10]}

a) While) after) tenv, fenv: {'one': [1, 1], 'a': [2, 10], 'b': [2, inf]}, {'one': [1, 1], 'b': [1, inf], 'a': [10, 10]}

a) Seq) env: {'one': [1, 1], 'b': [1, inf], 'a': [10, 10]}
```
즉, b는 while 시작 후 3번만 증가를 반복하면 env_widen에 의해 범위가 [(초기값),inf]이 된다. 
>만약 while 시작 후 4번째 반복부터 b를 감소시키면, a_interp에서 while 이후 b의 범위는 [(초기값),inf]이면서 interp에서 실제 while 실행 시 b가 음수가 되도록 만들 수 있다.
------------------------------
## Exploit scenario
```python
a = 10
b = 10

while a<50:
    a += 1         # a: [11,inf]
    if a<20:
        b += 1     # b: [11,inf]
    else:
        b -= 2     # b: [8,inf]
    
print(b)           # b: [8,inf], 10+1*9-2*31 = -43
```
*a와 b의 infimum이 초기값과 다른 이유는 if문 때문이다.*

-----------------------
## Exploit code
a=10;b=10;one=1;two=2;[a<50{a=a+one;a<20?{b=b+one}:{b=b-two}}];!b
```
hanch@ubuntu:~/Documents/cykor/codegate2020/verifier$ python3 prob.py 
> a=10;b=10;one=1;two=2;[a<50{a=a+one;a<20?{b=b+one}:{b=b-two}}];!b
...
a) While) after) tenv: {'b': [11, 14], 'one': [1, 1], 'two': [2, 2], 'a': [11, 14]}

after widen: {'b': [10, inf], 'one': [1, 1], 'two': [2, 2], 'a': [10, inf]}

a) While) after) tenv: {'b': [10, inf], 'one': [1, 1], 'two': [2, 2], 'a': [10, 49]}
a) Assign) env: {'b': [10, inf], 'one': [1, 1], 'two': [2, 2], 'a': [11, 50]}
a) if) tenv, fenv: {'b': [10, inf], 'one': [1, 1], 'two': [2, 2], 'a': [11, 19]}, {'b': [10, inf], 'one': [1, 1], 'two': [2, 2], 'a': [20, 50]}
a) Assign) env: {'b': [11, inf], 'one': [1, 1], 'two': [2, 2], 'a': [11, 19]}
a) Assign) env: {'b': [8, inf], 'one': [1, 1], 'two': [2, 2], 'a': [20, 50]}
a) Seq) env: {'b': [8, inf], 'one': [1, 1], 'two': [2, 2], 'a': [11, 50]}
a) While) after) tenv: {'b': [8, inf], 'one': [1, 1], 'two': [2, 2], 'a': [11, 50]}
a) While) after) fenv: {'b': [8, inf], 'one': [1, 1], 'two': [2, 2], 'a': [50, 50]}

a) While) after) tenv, fenv: {'b': [8, inf], 'one': [1, 1], 'two': [2, 2], 'a': [11, 50]}, {'b': [8, inf], 'one': [1, 1], 'two': [2, 2], 'a': [50, 50]}

a) print) env: {'b': [8, inf], 'one': [1, 1], 'two': [2, 2], 'a': [50, 50]}
a) print) a_val: [8, inf]
a) print) a_val.infimum: 8
a) Seq) env: {'b': [8, inf], 'one': [1, 1], 'two': [2, 2], 'a': [50, 50]}
...
...
if) cond: False
Assign) env: {'a': 50, 'b': -43, 'one': 1, 'two': 2}
Seq) env: {'a': 50, 'b': -43, 'one': 1, 'two': 2}
While) env: {'a': 50, 'b': -43, 'one': 1, 'two': 2}
value: -43
Error: [Errno 2] No such file or directory: './flag'
```
>로컬에서 실행 결과, 의도한대로 a_interp 단계에서 b의 최소값은 8이고, interp 단계에서 음수 -43이 출력되었다. 그리고 ./flag를 읽어오려는 모습을 확인할 수 있다.
