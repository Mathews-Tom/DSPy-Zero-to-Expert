export interface Objective {
  id: number;
  text: string;
  done: boolean;
}

export interface Stage {
  id: number;
  title: string;
  level: string;
  duration: string;
  description: string;
  objectives: Objective[];
  notes: string;
  complete: boolean;
}
