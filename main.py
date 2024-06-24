from dataclasses import dataclass, field
from functools import partial, singledispatch
from typing import Any, Callable, NamedTuple, Self
import numpy as np
from uuid import NAMESPACE_X500, UUID, uuid4, uuid5


@dataclass(frozen=True)
class Fmap:
    def fmap[A, B](self: Any, f: Callable[[A], B]) -> Any:
        # return type(self)(*map(f, self))
        raise NotImplementedError

@dataclass(frozen=True)
class NOT[A](Fmap):
    x: A
    def fmap[B](self: "NOT[A]", f: Callable[[A], B]) -> "NOT[B]":
        return NOT(f(self.x))

@dataclass(frozen=True)
class AND[A](Fmap):
    x: A
    y: A
    def fmap[B](self: "AND[A]", f: Callable[[A], B]) -> "AND[B]":
        return AND(f(self.x), f(self.y))

@dataclass(frozen=True)
class OR[A](Fmap):
    x: A
    y: A
    def fmap[B](self: "OR[A]", f: Callable[[A], B]) -> "OR[B]":
        return OR(f(self.x), f(self.y))

@dataclass(frozen=True)
class Input[A](Fmap):
    x: A
    def fmap[B](self: "Input[A]", f: Callable[[A], B]) -> "Input[B]":
        return Input(f(self.x))
    def __repr__(self) -> str:
        return repr(self.x)

type RecGates[A] = (
    Input[A]
    | NOT[RecGates[A]]
    | AND[RecGates[A]]
    | OR [RecGates[A]]
)

type Gates[A] = Input[A] | NOT[A] | AND[A] | OR[A]

def XOR[A](x: RecGates[A], y: RecGates[A]) -> RecGates[A]:
    return OR(AND(x, NOT(y)), AND(NOT(x), y))

def EQ[A](x: RecGates[A], y: RecGates[A]) -> RecGates[A]:
    return AND(OR(x, NOT(y)), OR(NOT(x), y))

def fold[A](f_algebra: Callable[[Gates[A]], A], g: RecGates[A]) -> A:
    if isinstance(g, Input):
        return f_algebra(g)
    else:
        return f_algebra(g.fmap(partial(fold, f_algebra)))

def map_inputs[A, B](f: Callable[[A], B], g: RecGates[A]) -> RecGates[B]:
    if isinstance(g, Input):
        return g.fmap(f)
    else:
        return g.fmap(partial(map_inputs, f))

def boolean_algebra(g: Gates[bool]) -> bool:
    match g:
        case Input(x):
            return x
        case NOT(x):
            return not x
        case AND(x, y):
            return x and y
        case OR(x, y):
            return x or y


def float_algebra(g: Gates[float]) -> float:
    match g:
        case Input(x):
            return x
        case NOT(x):
            return 1 - x
        case AND(x, y):
            return x * y
        case OR(x, y):
            return 1 - (1 - x) * (1 - y)

def derivative_algebra(g: Gates[float]) -> Any:
    match g:
        case Input(x):
            return 1
        case NOT(x):
            return -1
        case AND(x, y):
            return (y, x)
        case OR(x, y):
            return (1-y, 1-x)

def string_algebra(g: Gates[str]) -> str:
    match g:
        case Input(x):
            return x
        case NOT(x):
            return f"!({x})"
        case AND(x, y):
            return f"({x}&{y})"
        case OR(x, y):
            return f"({x}|{y})"


def from_uuids(*uuid: UUID) -> UUID:
    return uuid5(NAMESPACE_X500, "".join(map(str, uuid)))

@dataclass(frozen=True)
class Uniq[A](Fmap):
    unwrapped: A
    uuid: UUID = field(default_factory=uuid4)

    def fmap[B](self: "Uniq[A]", f: Callable[[A], B]) -> "Uniq[B]":
        return Uniq(f(self.unwrapped), from_uuids(self.uuid))
    
    def __repr__(self) -> str:
        return f"{self.unwrapped}#{str(self.uuid)[:4]}"


def uniquify_algebra[A](
        algebra: Callable[[Gates[A]], A],
        ) -> Callable[[Gates[Uniq[A]]], Uniq[A]]:
    def f(g: Gates[Uniq[A]]) -> Uniq[A]:
        uuids = (g.__getattribute__(field).uuid for field in g.__dataclass_fields__)
        unlifted = algebra(g.fmap(lambda x: x.unwrapped))
        return Uniq(unlifted, from_uuids(*uuids))
    return f

def example[A](x: A, y: A, z: A) -> RecGates[A]:
    x_input, y_input, z_input = Input(x), Input(y), Input(z)
    return EQ(AND(x_input,OR(AND(x_input, y_input), z_input)), NOT(z_input))

def bool_to_float(b: bool) -> float:
    return 1.0 if b else 0.0

def main() -> None:
    example_gates: RecGates[bool] = example(False, True, False)
    example_float_gates = map_inputs(bool_to_float, example_gates)
    print(f"Example: {example_float_gates}")
    print(f"String Algebra: {fold(string_algebra, map_inputs(str, example_gates))}")
    print(f"Boolean Algebra: {fold(boolean_algebra, example_gates)}")
    print(f"Float Algebra: {fold(float_algebra, map_inputs(bool_to_float, example_gates))}")
    print(f"Uniq inputs: {fold(string_algebra, map_inputs(lambda x: str(Uniq(x)), example_gates))}")
    print(f"Uniquified: {fold(uniquify_algebra(boolean_algebra), map_inputs(Uniq, example_gates))}")
    # print(f"Uniquified: {fold(string_algebra, example_uniq_gates)}")
    # print(f"Derivative Algebra: {fold(derivative_algebra, example_float_gates)}")

if __name__ == "__main__":
    main()