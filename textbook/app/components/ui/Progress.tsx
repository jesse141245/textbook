export function Progress({ value }: { value: number }) {
  return (
    <div className="w-full bg-gray-200 rounded-full h-3">
      <div className="bg-blue-500 h-3 rounded-full" style={{ width: `${value}%` }}></div>
    </div>
  );
}
