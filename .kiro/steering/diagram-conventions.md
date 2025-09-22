# Diagram Conventions and Standards

## Mermaid Diagram Orientation

**All Mermaid diagrams should preferably use vertical orientation (top-to-bottom) for consistency and readability.**

### Preferred Diagram Types and Orientations

#### Flowcharts
```mermaid
flowchart TD
    A[Start] --> B[Process]
    B --> C[Decision]
    C -->|Yes| D[Action]
    C -->|No| E[Alternative]
```

Use `TD` (Top Down) or `TB` (Top to Bottom) instead of `LR` (Left to Right).

#### State Diagrams
```mermaid
stateDiagram-v2
    [*] --> State1
    State1 --> State2
    State2 --> State3
    State3 --> [*]
```

State diagrams naturally flow vertically and should maintain this orientation.

#### Sequence Diagrams
```mermaid
sequenceDiagram
    participant A as Client
    participant B as Server
    A->>B: Request
    B->>A: Response
```

Sequence diagrams are inherently vertical and should remain so.

#### Class Diagrams
```mermaid
classDiagram
    class Animal {
        +String name
        +makeSound()
    }
    class Dog {
        +bark()
    }
    Animal <|-- Dog
```

Class diagrams should use vertical inheritance relationships where possible.

### Rationale

1. **Consistency**: Vertical orientation provides a consistent reading pattern across all documentation
2. **Readability**: Top-to-bottom flow matches natural reading patterns
3. **Mobile-friendly**: Vertical diagrams work better on mobile devices and narrow screens
4. **Print-friendly**: Vertical orientation fits better on standard page formats
5. **GitHub compatibility**: Vertical diagrams render more consistently in GitHub's Markdown viewer

### Exceptions

Horizontal orientation may be used only when:
- The diagram becomes too tall and loses readability
- The logical flow is inherently horizontal (e.g., timeline diagrams)
- Space constraints require horizontal layout

In such cases, document the reason for the exception in a comment above the diagram.

### Implementation Guidelines

- Always start flowcharts with `flowchart TD` or `flowchart TB`
- Use vertical node arrangements in complex diagrams
- Group related elements vertically rather than horizontally
- Consider breaking large horizontal diagrams into multiple vertical sections
- Use subgraphs to maintain vertical organization in complex diagrams

### Example: Converting Horizontal to Vertical

❌ **Avoid (Horizontal)**:
```mermaid
flowchart LR
    A --> B --> C --> D --> E
```

✅ **Prefer (Vertical)**:
```mermaid
flowchart TD
    A --> B
    B --> C
    C --> D
    D --> E
```

Or for complex flows:
```mermaid
flowchart TD
    A[Input] --> B[Process 1]
    B --> C[Process 2]
    C --> D[Decision]
    D -->|Yes| E[Success Path]
    D -->|No| F[Error Path]
    E --> G[Output]
    F --> G
```

This convention ensures all diagrams in the project maintain a consistent, readable, and professional appearance.